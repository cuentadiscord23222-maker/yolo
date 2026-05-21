/*
 * YOLO AI Detector - Full GPU pipeline, auto-start
 * DXGI Desktop Duplication → GPU resize → DirectML → D3D11 overlay
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>
#include <commctrl.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <math.h>
#include <atomic>
#include <vector>
#include <mutex>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "winmm.lib")

#include "onnxruntime_c_api.h"
#pragma comment(lib, "onnxruntime.lib")

static int SCREEN_W = 0, SCREEN_H = 0;
static HWND hOverlay = NULL;
static HWND hPanel = NULL;
static HWND hFovShape = NULL;
static HWND hTrackLMB = NULL, hTrackRMB = NULL, hSTrack = NULL, hTrackSpeedV = NULL, hShowBoxes = NULL;

/* D3D11 */
static ID3D11Device* g_dev = NULL;
static ID3D11DeviceContext* g_ctx = NULL;
static IDXGISwapChain* g_swap = NULL;
static ID3D11RenderTargetView* g_rtv = NULL;
static ID3D11VertexShader* g_vs = NULL;
static ID3D11PixelShader* g_ps = NULL;
static ID3D11InputLayout* g_il = NULL;
static ID3D11Buffer* g_vb = NULL;

/* DXGI Desktop Duplication */
static IDXGIOutputDuplication* g_dup = NULL;

/* GPU textures */
static ID3D11Texture2D* g_input_tex = NULL;
static ID3D11Texture2D* g_staging = NULL;
static ID3D11RenderTargetView* g_input_rtv = NULL;
static ID3D11Texture2D* g_cap_staging = NULL; /* persistent capture staging */

/* Resize shader */
static ID3D11VertexShader* g_rvs = NULL;
static ID3D11PixelShader* g_rps = NULL;
static ID3D11SamplerState* g_sampler = NULL;
static ID3D11ShaderResourceView* g_desk_srv = NULL;
static ID3D11Buffer* g_resize_cb = NULL;

/* State */
static std::atomic<bool> g_running{true};  /* auto-start */
static std::atomic<int> g_fov{200};
static std::atomic<float> g_conf{0.30f};
static std::atomic<int> g_maxfps{300};
static std::atomic<int> g_maxrfps{60};
static std::atomic<int> g_fov_shape{0}; /* 0=circle, 1=square */
static std::atomic<int> g_track_btn{0}; /* 0=none, 1=LMB, 2=RMB */
static std::atomic<int> g_track_speed{50}; /* 1-100, maps to 0.1-1.0 */
static std::atomic<bool> g_tracking{false};
static std::atomic<bool> g_show_boxes{true};

struct Det { int x1,y1,x2,y2,cls; float conf; };
static std::mutex g_mtx;
static std::vector<Det> g_dets;
static std::atomic<int> g_fps{0}, g_ifps{0}, g_rfps{0}, g_ndet{0};
static std::atomic<double> g_t_cap{0}, g_t_run{0}, g_t_total{0};
static std::mutex g_d3d_mtx;

/* Class skip: true = skip this class. Default: skip all except person (0) */
static bool g_skip_cls[80] = {false};
static void init_skip() {
    for(int i = 1; i < 80; i++) g_skip_cls[i] = true;
}

/* ONNX */
static const OrtApi* g_api = NULL;
static OrtEnv* g_env = NULL;
static OrtSession* g_sess = NULL;
static OrtMemoryInfo* g_mem = NULL;
static const int MSZ = 640;

static const char* CLS[80] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

static unsigned int COL[80];
static void init_col() {
    unsigned int s = 42;
    for (int i = 0; i < 80; i++) {
        s = s*1103515245+12345; int r=(s>>16)&0xFF;
        s = s*1103515245+12345; int g=(s>>16)&0xFF;
        s = s*1103515245+12345; int b=(s>>16)&0xFF;
        COL[i] = (r<<16)|(g<<8)|b;
    }
}

static double perf_now() {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart / freq.QuadPart;
}

static void track_target() {
    std::vector<Det> dets;
    { std::lock_guard<std::mutex> lk(g_mtx); dets = g_dets; }
    if(dets.empty()) return;

    int cx = SCREEN_W / 2;
    int cy = SCREEN_H / 2;

    int target_x = 0, target_y = 0;
    float best_dist = 1e9f;
    for(int i = 0; i < (int)dets.size(); i++) {
        int bcx = (dets[i].x1 + dets[i].x2) / 2;
        int bcy = (dets[i].y1 + dets[i].y2) / 2;
        float dx = (float)(bcx - cx);
        float dy = (float)(bcy - cy);
        float dist = sqrtf(dx*dx + dy*dy);
        if(dist < best_dist) {
            best_dist = dist;
            target_x = bcx;
            target_y = bcy;
        }
    }

    int dx = target_x - cx;
    int dy = target_y - cy;

    float spd = (float)g_track_speed.load();
    float factor = 0.05f + (spd - 1.0f) * (0.95f / 99.0f);

    int mx = (int)((float)dx * factor);
    int my = (int)((float)dy * factor);

    if(dx > 1 && mx == 0) mx = 1;
    else if(dx < -1 && mx == 0) mx = -1;
    if(dy > 1 && my == 0) my = 1;
    else if(dy < -1 && my == 0) my = -1;

    INPUT inp = {};
    inp.type = INPUT_MOUSE;
    inp.mi.dx = mx;
    inp.mi.dy = my;
    inp.mi.dwFlags = MOUSEEVENTF_MOVE;
    SendInput(1, &inp, sizeof(INPUT));
}

static DWORD WINAPI track_thread(LPVOID) {
    float vx = (float)(SCREEN_W / 2);
    float vy = (float)(SCREEN_H / 2);

    int last_tx = 0, last_ty = 0;
    bool has_lock = false;

    while(g_running) {
        int btn = g_track_btn.load();
        bool pressed = false;
        if(btn == 1 && (GetAsyncKeyState(VK_LBUTTON) & 0x8000)) pressed = true;
        if(btn == 2 && (GetAsyncKeyState(VK_RBUTTON) & 0x8000)) pressed = true;

        if(pressed) {
            std::vector<Det> dets;
            { std::lock_guard<std::mutex> lk(g_mtx); dets = g_dets; }

            if(dets.empty()) {
                Sleep(1);
                continue;
            }

            int target_x = 0, target_y = 0;

            if(has_lock && (last_tx != 0 || last_ty != 0)) {
                /* Find detection closest to the LAST known target position */
                float best_dist = 1e9f;
                for(int i = 0; i < (int)dets.size(); i++) {
                    int bcx = (dets[i].x1 + dets[i].x2) / 2;
                    int bcy = (dets[i].y1 + dets[i].y2) / 2;
                    float dx = (float)(bcx - last_tx);
                    float dy = (float)(bcy - last_ty);
                    float dist = sqrtf(dx*dx + dy*dy);
                    if(dist < best_dist) {
                        best_dist = dist;
                        target_x = bcx;
                        target_y = bcy;
                    }
                }
            } else {
                /* Choose target closest to SCREEN CENTER (not cursor) */
                float best_dist = 1e9f;
                int cx = SCREEN_W / 2, cy = SCREEN_H / 2;
                for(int i = 0; i < (int)dets.size(); i++) {
                    int bcx = (dets[i].x1 + dets[i].x2) / 2;
                    int bcy = (dets[i].y1 + dets[i].y2) / 2;
                    float dx = (float)(bcx - cx);
                    float dy = (float)(bcy - cy);
                    float dist = sqrtf(dx*dx + dy*dy);
                    if(dist < best_dist) {
                        best_dist = dist;
                        target_x = bcx;
                        target_y = bcy;
                    }
                }
            }

            /* Move VIRTUAL cursor toward target each frame */
            float spd = g_track_speed.load();
            float lerp = 0.02f + (spd - 1) * 0.98f / 99.0f;

            vx += (float)(target_x - (int)vx) * lerp;
            vy += (float)(target_y - (int)vy) * lerp;

            if(vx < 0) vx = 0;
            if(vy < 0) vy = 0;
            if(vx > SCREEN_W-1) vx = (float)(SCREEN_W-1);
            if(vy > SCREEN_H-1) vy = (float)(SCREEN_H-1);

            static float prev_vx = vx, prev_vy = vy;
            int mx = (int)(vx - prev_vx);
            int my = (int)(vy - prev_vy);

            if(mx != 0 || my != 0) {
                INPUT inp = {};
                inp.type = INPUT_MOUSE;
                inp.mi.dx = mx;
                inp.mi.dy = my;
                inp.mi.dwFlags = MOUSEEVENTF_MOVE;
                SendInput(1, &inp, sizeof(INPUT));
            }

            prev_vx = vx;
            prev_vy = vy;

            last_tx = target_x;
            last_ty = target_y;
            has_lock = true;
        } else {
            vx = (float)(SCREEN_W / 2);
            vy = (float)(SCREEN_H / 2);
            last_tx = 0;
            last_ty = 0;
            has_lock = false;
            Sleep(5);
        }
    }
    return 0;
}

/* ── Shaders ── */
static const char* g_resize_vs_code = R"(
struct VSOut { float4 Pos : SV_POSITION; float2 UV : TEXCOORD; };
VSOut main(uint id : SV_VertexID) {
    float2 pos, uv;
    switch(id) {
        case 0: pos=float2(-1, 1); uv=float2(0,0); break;
        case 1: pos=float2( 1, 1); uv=float2(1,0); break;
        case 2: pos=float2(-1,-1); uv=float2(0,1); break;
        case 3: pos=float2(-1,-1); uv=float2(0,1); break;
        case 4: pos=float2( 1, 1); uv=float2(1,0); break;
        case 5: pos=float2( 1,-1); uv=float2(1,1); break;
    }
    VSOut o; o.Pos=float4(pos,0,1); o.UV=uv; return o;
})";

static const char* g_resize_ps_code = R"(
Texture2D tex : register(t0);
SamplerState samp : register(s0);
float4 main(float2 uv : TEXCOORD) : SV_TARGET {
    return tex.Sample(samp, uv);
})";

static const char* g_vs_code = R"(
struct VSIn { float2 Pos : POSITION; float4 Col : COLOR; };
struct VSOut { float4 Pos : SV_POSITION; float4 Col : COLOR; };
VSOut main(VSIn i) { VSOut o; o.Pos=float4(i.Pos,0,1); o.Col=i.Col; return o; })";

static const char* g_ps_code = R"(
struct PSIn { float4 Pos : SV_POSITION; float4 Col : COLOR; };
float4 main(PSIn i) : SV_TARGET { return i.Col; })";

struct Vertex { float x,y,r,g,b,a; };

/* ── GPU capture + CPU resize ── */
static std::vector<float> gpu_capture(int cx, int cy, int r) {
    int cap_sz = 640;
    int cap_x1 = cx - cap_sz/2, cap_y1 = cy - cap_sz/2;

    IDXGIResource* res = NULL;
    DXGI_OUTDUPL_FRAME_INFO fi;

    HRESULT hr = g_dup->AcquireNextFrame(16, &fi, &res);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) return std::vector<float>();
    if (FAILED(hr) || !res) {
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            g_dup->Release();
            IDXGIDevice* dxgi_dev = NULL;
            g_dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_dev);
            IDXGIAdapter* adapter = NULL;
            dxgi_dev->GetAdapter(&adapter);
            IDXGIOutput* output = NULL;
            adapter->EnumOutputs(0, &output);
            IDXGIOutput1* out1 = NULL;
            output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1);
            out1->DuplicateOutput(g_dev, &g_dup);
            out1->Release(); output->Release(); adapter->Release(); dxgi_dev->Release();
            hr = g_dup->AcquireNextFrame(50, &fi, &res);
            if (FAILED(hr) || !res) return std::vector<float>();
        } else return std::vector<float>();
    }

    ID3D11Texture2D* desk = NULL;
    res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&desk);
    res->Release();

    D3D11_TEXTURE2D_DESC dtd;
    desk->GetDesc(&dtd);

    g_ctx->CopyResource(g_cap_staging, desk);
    g_ctx->Flush();
    desk->Release();
    g_dup->ReleaseFrame();

    D3D11_MAPPED_SUBRESOURCE dms;
    HRESULT hr_map = g_ctx->Map(g_cap_staging, 0, D3D11_MAP_READ, 0, &dms);
    if(FAILED(hr_map)) return std::vector<float>();

    uint8_t* dsrc = (uint8_t*)dms.pData;
    int dpitch = dms.RowPitch;
    if(!dsrc || dpitch <= 0) { g_ctx->Unmap(g_cap_staging, 0); return std::vector<float>(); }

    std::vector<float> ten(3 * MSZ * MSZ);
    float *R = &ten[0], *G = &ten[MSZ*MSZ], *B = &ten[MSZ*MSZ*2];
    float inv = 0.0039215686f;

    for (int ty = 0; ty < MSZ; ty++) {
        int sy = cap_y1 + ty * cap_sz / MSZ;
        if(sy < 0) sy = 0; if(sy >= (int)dtd.Height) sy = (int)dtd.Height - 1;
        int row = sy * dpitch;
        for (int tx = 0; tx < MSZ; tx++) {
            int sx = cap_x1 + tx * cap_sz / MSZ;
            if(sx < 0) sx = 0; if(sx >= (int)dtd.Width) sx = (int)dtd.Width - 1;
            int idx = row + sx * 4;
            R[ty*MSZ+tx] = dsrc[idx+2] * inv;
            G[ty*MSZ+tx] = dsrc[idx+1] * inv;
            B[ty*MSZ+tx] = dsrc[idx+0] * inv;
        }
    }

    g_ctx->Unmap(g_cap_staging, 0);
    return ten;
}

/* ── Parse output ── */
static std::vector<Det> parse(float* out, int cx, int cy, int fov, float ct) {
    std::vector<Det> d;
    /* Capture is always 640x640 centered on screen */
    int cap_sz = 640;
    int cap_x1 = cx - cap_sz/2, cap_y1 = cy - cap_sz/2;
    float fov_f = (float)fov;

    for (int i = 0; i < 300; i++) {
        float* r = &out[i*6];
        if (r[4] < ct) continue;
        int c = (int)r[5]; if(c<0||c>=80) continue;
        if(g_skip_cls[c]) continue;

        /* Format: x1, y1, x2, y2 in 640x640 input space → map to screen */
        int bx1 = (int)r[0] + cap_x1;
        int by1 = (int)r[1] + cap_y1;
        int bx2 = (int)r[2] + cap_x1;
        int by2 = (int)r[3] + cap_y1;

        int bw = bx2 - bx1, bh = by2 - by1;
        if (bw <= 0 || bh <= 0) continue;

        /* FOV check: any pixel of box inside FOV = detect */
        int shape = g_fov_shape.load();
        if(shape == 0) {
            /* Circle: closest point on box to center must be within radius */
            int closest_x = (cx < bx1) ? bx1 : (cx > bx2) ? bx2 : cx;
            int closest_y = (cy < by1) ? by1 : (cy > by2) ? by2 : cy;
            int dx = closest_x - cx, dy = closest_y - cy;
            if(dx*dx + dy*dy > fov_f * fov_f) continue;
        } else {
            /* Square: box must overlap with [cx-fov, cy-fov, cx+fov, cy+fov] */
            int sx1 = cx - (int)fov_f, sy1 = cy - (int)fov_f;
            int sx2 = cx + (int)fov_f, sy2 = cy + (int)fov_f;
            if(bx2 < sx1 || bx1 > sx2 || by2 < sy1 || by1 > sy2) continue;
        }

        d.push_back({bx1,by1,bx2,by2,c,r[4]});
    }
    return d;
}

/* ── Inference thread ── */
static void render();

static DWORD WINAPI main_thread(LPVOID) {
    int cx = SCREEN_W/2, cy = SCREEN_H/2;
    int ic = 0, rc = 0;
    double it = perf_now(), rt = perf_now();
    const int64_t ishape[] = {1, 3, MSZ, MSZ};
    const char* iname[] = {"images"}, *oname[] = {"output0"};
    std::vector<float> obuf(300 * 6);
    double last_render = 0, last_infer = 0;

    while(g_running) {
        double now = perf_now();
        int maxfps = g_maxfps.load();
        int maxrfps = g_maxrfps.load();
        int fov = g_fov.load();

        bool need_infer = (maxfps <= 0) || (now - last_infer >= 1.0 / maxfps);
        bool need_render = (maxrfps <= 0) || (now - last_render >= 1.0 / maxrfps);

        if(need_infer) {
            double t_start = perf_now();
            auto ten = gpu_capture(cx, cy, fov);
            double t_cap = perf_now() - t_start;

            if(!ten.empty()) {
                OrtValue* itensor = NULL;
                g_api->CreateTensorWithDataAsOrtValue(g_mem, ten.data(), ten.size()*sizeof(float),
                    ishape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &itensor);

                double t0 = perf_now();
                OrtValue* otensor = NULL;
                g_api->Run(g_sess, NULL, iname, (const OrtValue* const*)&itensor, 1, oname, 1, &otensor);
                double t_run = perf_now() - t0;
                g_api->ReleaseValue(itensor);

                float* odata = NULL;
                g_api->GetTensorMutableData(otensor, (void**)&odata);
                memcpy(obuf.data(), odata, obuf.size() * sizeof(float));
                g_api->ReleaseValue(otensor);

                auto d = parse(obuf.data(), cx, cy, fov, g_conf.load());
                double total = perf_now() - t_start;

                { std::lock_guard<std::mutex> lk(g_mtx); g_dets = std::move(d); g_ndet = (int)g_dets.size(); }

                if(g_tracking.load()) track_target();

                g_t_cap = t_cap * 1000;
                g_t_run = t_run * 1000;
                g_t_total = total * 1000;

                ic++;
                if(now-it >= 1.0) { g_ifps = ic; ic = 0; it = now; }
            }
            last_infer = now;
        }

        if(need_render) {
            render();
            rc++;
            last_render = now;
        }

        if(now-rt >= 1.0) { g_rfps = rc; rc = 0; rt = now; }

        if(!need_infer && !need_render) {
            Sleep(1);
        }
    }
    return 0;
}

static bool compile_shaders() {
    ID3DBlob* blob=NULL;
    D3DCompile(g_resize_vs_code, strlen(g_resize_vs_code), NULL, NULL, NULL, "main", "vs_5_0", 0, 0, &blob, NULL);
    g_dev->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), NULL, &g_rvs);
    blob->Release();

    D3DCompile(g_resize_ps_code, strlen(g_resize_ps_code), NULL, NULL, NULL, "main", "ps_5_0", 0, 0, &blob, NULL);
    g_dev->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), NULL, &g_rps);
    blob->Release();

    D3D11_SAMPLER_DESC sd = {};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    g_dev->CreateSamplerState(&sd, &g_sampler);

    D3DCompile(g_vs_code, strlen(g_vs_code), NULL, NULL, NULL, "main", "vs_5_0", 0, 0, &blob, NULL);
    g_dev->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), NULL, &g_vs);

    D3D11_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION",0,DXGI_FORMAT_R32G32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,8,D3D11_INPUT_PER_VERTEX_DATA,0},
    };
    g_dev->CreateInputLayout(layout, 2, blob->GetBufferPointer(), blob->GetBufferSize(), &g_il);
    blob->Release();

    D3DCompile(g_ps_code, strlen(g_ps_code), NULL, NULL, NULL, "main", "ps_5_0", 0, 0, &blob, NULL);
    g_dev->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), NULL, &g_ps);
    blob->Release();

    D3D11_BUFFER_DESC bd = {};
    bd.Usage = D3D11_USAGE_DYNAMIC; bd.ByteWidth = sizeof(Vertex)*10000;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    g_dev->CreateBuffer(&bd, NULL, &g_vb);

    return true;
}

static bool init_d3d(HWND hwnd) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2; sd.BufferDesc.Width = SCREEN_W; sd.BufferDesc.Height = SCREEN_H;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60; sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd; sd.SampleDesc.Count = 1; sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    D3D_FEATURE_LEVEL lv[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
    if(FAILED(D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0,
        lv, 2, D3D11_SDK_VERSION, &sd, &g_swap, &g_dev, NULL, &g_ctx))) return false;

    ID3D11Texture2D* bb = NULL;
    g_swap->GetBuffer(0, IID_PPV_ARGS(&bb));
    g_dev->CreateRenderTargetView(bb, NULL, &g_rtv);
    bb->Release();

    /* GPU textures */
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = MSZ; td.Height = MSZ; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_RENDER_TARGET;
    g_dev->CreateTexture2D(&td, NULL, &g_input_tex);
    g_dev->CreateRenderTargetView(g_input_tex, NULL, &g_input_rtv);

    td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    td.MiscFlags = 0;
    if(FAILED(g_dev->CreateTexture2D(&td, NULL, &g_staging))) {
        MessageBoxA(NULL, "Failed to create staging texture", "Error", MB_ICONERROR);
        return false;
    }

    /* Persistent capture staging texture (full screen) */
    D3D11_TEXTURE2D_DESC td_cap = {};
    td_cap.Width = SCREEN_W; td_cap.Height = SCREEN_H; td_cap.MipLevels = 1; td_cap.ArraySize = 1;
    td_cap.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td_cap.SampleDesc.Count = 1;
    td_cap.Usage = D3D11_USAGE_STAGING; td_cap.BindFlags = 0;
    td_cap.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td_cap.MiscFlags = 0;
    g_dev->CreateTexture2D(&td_cap, NULL, &g_cap_staging);

    /* Desktop Duplication */
    IDXGIDevice* dxgi_dev = NULL;
    g_dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_dev);
    IDXGIAdapter* adapter = NULL;
    dxgi_dev->GetAdapter(&adapter);
    IDXGIOutput* output = NULL;
    adapter->EnumOutputs(0, &output);
    IDXGIOutput1* out1 = NULL;
    output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1);
    out1->DuplicateOutput(g_dev, &g_dup);
    out1->Release(); output->Release(); adapter->Release(); dxgi_dev->Release();

    return compile_shaders();
}

static void cleanup_d3d() {
    if(g_dup) g_dup->Release();
    if(g_input_tex) g_input_tex->Release();
    if(g_staging) g_staging->Release();
    if(g_input_rtv) g_input_rtv->Release();
    if(g_rvs) g_rvs->Release();
    if(g_rps) g_rps->Release();
    if(g_sampler) g_sampler->Release();
    if(g_rtv) g_rtv->Release();
    if(g_swap) g_swap->Release();
    if(g_ctx) g_ctx->Release();
    if(g_dev) g_dev->Release();
    if(g_vs) g_vs->Release();
    if(g_ps) g_ps->Release();
    if(g_il) g_il->Release();
    if(g_vb) g_vb->Release();
    if(g_cap_staging) g_cap_staging->Release();
}

/* ── Drawing helpers ── */
static void add_line(std::vector<Vertex>& v, float x1,float y1,float x2,float y2,
                      float r,float g,float b) {
    v.push_back({x1,y1,r,g,b,1});
    v.push_back({x2,y2,r,g,b,1});
}

static void add_circle(std::vector<Vertex>& v, float cx,float cy,float rad,
                        float r,float g,float b) {
    for(int i=0;i<64;i++) {
        float a1=6.28318f*i/64, a2=6.28318f*(i+1)/64;
        add_line(v, cx+rad*cosf(a1), cy+rad*sinf(a1), cx+rad*cosf(a2), cy+rad*sinf(a2), r,g,b);
    }
}

static void add_circle_ar(std::vector<Vertex>& v, float cx,float cy,float rad,
                           float aspect, float r,float g,float b) {
    float rx = rad / aspect;
    for(int i=0;i<64;i++) {
        float a1=6.28318f*i/64, a2=6.28318f*(i+1)/64;
        add_line(v, cx+rx*cosf(a1), cy+rad*sinf(a1), cx+rx*cosf(a2), cy+rad*sinf(a2), r,g,b);
    }
}

static void add_square(std::vector<Vertex>& v, float cx,float cy,float rad, float aspect, float r,float g,float b) {
    float rx = rad / aspect;
    float x1=cx-rx, y1=cy-rad, x2=cx+rx, y2=cy+rad;
    add_line(v,x1,y1,x2,y1, r,g,b);
    add_line(v,x2,y1,x2,y2, r,g,b);
    add_line(v,x2,y2,x1,y2, r,g,b);
    add_line(v,x1,y2,x1,y1, r,g,b);
}

static void render() {
    static int rc = 0;
    static double rt = perf_now();

    if(!g_ctx || !g_rtv) return;

    int cx = SCREEN_W/2, cy = SCREEN_H/2, r = g_fov.load();
    std::vector<Vertex> verts;
    verts.reserve(5000);

    float hw = (float)SCREEN_W, hh = (float)SCREEN_H;
    float cx_n = cx/hw*2.0f - 1.0f, cy_n = -(cy/hh*2.0f - 1.0f);
    float r_n = (float)r / hh * 2.0f;
    float aspect = hw / hh;

    if(g_fov_shape.load() == 0) {
        add_circle_ar(verts, cx_n, cy_n, r_n, aspect, 0,1,1);
    } else {
        add_square(verts, cx_n, cy_n, r_n, aspect, 0,1,1);
    }

    if(g_show_boxes.load()) {
        std::lock_guard<std::mutex> lk(g_mtx);
        for(auto& d : g_dets) {
            unsigned int c = COL[d.cls];
            float cr = ((c>>16)&0xFF)/255.0f, cg = ((c>>8)&0xFF)/255.0f, cb = (c&0xFF)/255.0f;
            float x1=d.x1/hw*2-1, y1=-(d.y1/hh*2-1);
            float x2=d.x2/hw*2-1, y2=-(d.y2/hh*2-1);
            add_line(verts,x1,y1,x2,y1, cr,cg,cb);
            add_line(verts,x2,y1,x2,y2, cr,cg,cb);
            add_line(verts,x2,y2,x1,y2, cr,cg,cb);
            add_line(verts,x1,y2,x1,y1, cr,cg,cb);
        }
    }

    float cc[4] = {0,0,0,0};
    g_ctx->ClearRenderTargetView(g_rtv, cc);

    if(!verts.empty()) {
        D3D11_MAPPED_SUBRESOURCE ms;
        g_ctx->Map(g_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms);
        memcpy(ms.pData, verts.data(), verts.size()*sizeof(Vertex));
        g_ctx->Unmap(g_vb, 0);

        UINT stride = sizeof(Vertex), offset = 0;
        g_ctx->IASetVertexBuffers(0, 1, &g_vb, &stride, &offset);
        g_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
        g_ctx->IASetInputLayout(g_il);
        g_ctx->VSSetShader(g_vs, NULL, 0);
        g_ctx->PSSetShader(g_ps, NULL, 0);
        g_ctx->OMSetRenderTargets(1, &g_rtv, NULL);

        D3D11_VIEWPORT vp = {0, 0, (float)SCREEN_W, (float)SCREEN_H, 0, 1};
        g_ctx->RSSetViewports(1, &vp);
        g_ctx->Draw((UINT)verts.size(), 0);
    }

    g_swap->Present(0, 0);
}

/* ── Panel ── */
static HWND hIfps,hRfps,hDet,hSFps,hSRfps,hSFOV,hSConf,hFpsV,hRfpsV,hFovV,hConfV,hDebug;

static LRESULT CALLBACK PanelProc(HWND hwnd,UINT msg,WPARAM wp,LPARAM lp) {
    if(msg==WM_HSCROLL) {
        if((HWND)lp==hSFps) { int v=(int)SendMessage(hSFps,TBM_GETPOS,0,0); g_maxfps=v; char b[8]; snprintf(b,sizeof(b),"%d",v); SetWindowTextA(hFpsV,b); }
        if((HWND)lp==hSRfps) { int v=(int)SendMessage(hSRfps,TBM_GETPOS,0,0); g_maxrfps=v; char b[8]; snprintf(b,sizeof(b),"%d",v); SetWindowTextA(hRfpsV,b); }
        if((HWND)lp==hSFOV) { int v=(int)SendMessage(hSFOV,TBM_GETPOS,0,0); g_fov=v; char b[8]; snprintf(b,sizeof(b),"%d",v); SetWindowTextA(hFovV,b); }
        if((HWND)lp==hSConf) { int v=(int)SendMessage(hSConf,TBM_GETPOS,0,0); float c=v/100.0f; g_conf=c; char b[8]; snprintf(b,sizeof(b),"%.2f",c); SetWindowTextA(hConfV,b); }
        if((HWND)lp==hSTrack) { int v=(int)SendMessage(hSTrack,TBM_GETPOS,0,0); g_track_speed=v; char b[16]; snprintf(b,sizeof(b),"%.2f", 0.1f + (v-1)*0.9f/99.0f); SetWindowTextA(hTrackSpeedV,b); }
        return 0;
    }
    if(msg==WM_COMMAND && HIWORD(wp)==BN_CLICKED && (HWND)lp==hShowBoxes) {
        g_show_boxes = !g_show_boxes.load();
        SetWindowTextA(hShowBoxes, g_show_boxes.load() ? "ON" : "OFF");
        return 0;
    }
    if(msg==WM_COMMAND && HIWORD(wp)==BN_CLICKED && (HWND)lp==hFovShape) {
        int cur = g_fov_shape.load();
        int next = cur == 0 ? 1 : 0;
        g_fov_shape = next;
        SetWindowTextA(hFovShape, next == 0 ? "Circle" : "Square");
        return 0;
    }
    if(msg==WM_COMMAND && HIWORD(wp)==BN_CLICKED && (HWND)lp==hTrackLMB) {
        int cur = g_track_btn.load();
        g_track_btn = (cur == 1) ? 0 : 1;
        SetWindowTextA(hTrackLMB, g_track_btn.load() == 1 ? "LMB [ON]" : "LMB");
        if(g_track_btn.load() != 1) SetWindowTextA(hTrackRMB, "RMB");
        return 0;
    }
    if(msg==WM_COMMAND && HIWORD(wp)==BN_CLICKED && (HWND)lp==hTrackRMB) {
        int cur = g_track_btn.load();
        g_track_btn = (cur == 2) ? 0 : 2;
        SetWindowTextA(hTrackRMB, g_track_btn.load() == 2 ? "RMB [ON]" : "RMB");
        if(g_track_btn.load() != 2) SetWindowTextA(hTrackLMB, "LMB");
        return 0;
    }
    if(msg==WM_TIMER) {
        /* Sync sliders to current values */
        int v_fps = (int)SendMessage(hSFps,TBM_GETPOS,0,0);
        if(g_maxfps.load() != v_fps) { g_maxfps = v_fps; }
        int v_rfps = (int)SendMessage(hSRfps,TBM_GETPOS,0,0);
        if(g_maxrfps.load() != v_rfps) { g_maxrfps = v_rfps; }
        int v_fov = (int)SendMessage(hSFOV,TBM_GETPOS,0,0);
        if(g_fov.load() != v_fov) { g_fov = v_fov; }
        int v_conf = (int)SendMessage(hSConf,TBM_GETPOS,0,0);
        if(g_conf.load() != v_conf/100.0f) { g_conf = v_conf/100.0f; }
        int v_track = (int)SendMessage(hSTrack,TBM_GETPOS,0,0);
        if(g_track_speed.load() != v_track) { g_track_speed = v_track; }

        char b[64];
        snprintf(b,sizeof(b),"%d",g_ifps.load()); SetWindowTextA(hIfps,b);
        snprintf(b,sizeof(b),"%d",g_rfps.load()); SetWindowTextA(hRfps,b);
        snprintf(b,sizeof(b),"%d",g_ndet.load()); SetWindowTextA(hDet,b);
        {
            float spd = 0.1f + (g_track_speed.load()-1)*0.9f/99.0f;
            snprintf(b, sizeof(b),"%.2f",spd);
            SetWindowTextA(hTrackSpeedV,b);
        }
        if(hDebug) {
            snprintf(b, sizeof(b), "cap=%.0fms run=%.0fms total=%.0fms",
                g_t_cap.load(), g_t_run.load(), g_t_total.load());
            SetWindowTextA(hDebug, b);
        }
        return 0;
    }
    if(msg==WM_CLOSE) { ShowWindow(hwnd,SW_HIDE); return 0; }
    if(msg==WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProc(hwnd,msg,wp,lp);
}

int WINAPI WinMain(HINSTANCE hInst,HINSTANCE,LPSTR,int) {
    SetProcessDPIAware();
    timeBeginPeriod(1);

    SCREEN_W = GetSystemMetrics(SM_CXSCREEN);
    SCREEN_H = GetSystemMetrics(SM_CYSCREEN);
    init_col();
    init_skip();
    InitCommonControls();

    /* Debug log file */
    FILE* dbg = fopen("debug_log.txt", "w");
    if(dbg) {
        fprintf(dbg, "[YOLO Detector] Starting...\n");
        fprintf(dbg, "Screen: %dx%d\n", SCREEN_W, SCREEN_H);
        fflush(dbg);
    }

    g_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if(!g_api) { MessageBoxA(NULL,"ONNX Runtime API failed","Error",MB_ICONERROR); return 1; }

    g_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"yolo",&g_env);
    OrtSessionOptions* so = NULL;
    g_api->CreateSessionOptions(&so);
    g_api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);

    const char* prov = "DmlExecutionProvider";
    g_api->SessionOptionsAppendExecutionProvider(so, prov, NULL, NULL, 0);

    g_api->CreateSession(g_env, L"yolo26s.onnx", so, &g_sess);
    g_api->ReleaseSessionOptions(so);
    if(!g_sess) { MessageBoxA(NULL,"Failed to load yolo26s.onnx","Error",MB_ICONERROR); return 1; }

    g_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_mem);

    if(dbg) { fprintf(dbg, "ONNX model loaded. Starting D3D11...\n"); fflush(dbg); }

    /* Overlay */
    WNDCLASSEXA wc = {sizeof(WNDCLASSEXA),0,DefWindowProcA,0,0,hInst,NULL,LoadCursor(NULL,IDC_ARROW),NULL,NULL,"Overlay",NULL};
    RegisterClassExA(&wc);

    hOverlay = CreateWindowExA(WS_EX_TOPMOST|WS_EX_LAYERED|WS_EX_TRANSPARENT|WS_EX_TOOLWINDOW,
        "Overlay","",WS_POPUP,0,0,SCREEN_W,SCREEN_H,NULL,NULL,hInst,NULL);
    SetLayeredWindowAttributes(hOverlay, 0x000000, 255, LWA_COLORKEY);

    if(!init_d3d(hOverlay)) { MessageBoxA(NULL,"D3D11/DXGI failed","Error",MB_ICONERROR); return 1; }

    ShowWindow(hOverlay, SW_SHOW); UpdateWindow(hOverlay);

    /* Panel */
    WNDCLASSEXA wc2 = {sizeof(WNDCLASSEXA),0,PanelProc,0,0,hInst,NULL,LoadCursor(NULL,IDC_ARROW),
        (HBRUSH)(COLOR_BTNFACE+1),NULL,"Panel",NULL};
    RegisterClassExA(&wc2);

    hPanel = CreateWindowExA(WS_EX_TOPMOST|WS_EX_TOOLWINDOW,"Panel","YOLO Control",
        WS_CAPTION|WS_SYSMENU|WS_VISIBLE|WS_THICKFRAME,50,50,320,520,NULL,NULL,hInst,NULL);

    HFONT hf = CreateFontA(20,0,0,0,FW_BOLD,0,0,0,DEFAULT_CHARSET,0,0,DEFAULT_QUALITY,0,"Consolas");
    HFONT hm = CreateFontA(16,0,0,0,FW_BOLD,0,0,0,DEFAULT_CHARSET,0,0,DEFAULT_QUALITY,0,"Consolas");
    HFONT hs = CreateFontA(14,0,0,0,0,0,0,0,DEFAULT_CHARSET,0,0,DEFAULT_QUALITY,0,"Consolas");

    int y = 10;

    CreateWindowA("STATIC","Inference FPS:",WS_VISIBLE|WS_CHILD,15,y,120,20,hPanel,NULL,hInst,NULL);
    hIfps = CreateWindowA("STATIC","0",WS_VISIBLE|WS_CHILD|SS_RIGHT,140,y,60,25,hPanel,NULL,hInst,NULL);
    SendMessage(hIfps,WM_SETFONT,(WPARAM)hf,1); y+=28;

    CreateWindowA("STATIC","Render FPS:",WS_VISIBLE|WS_CHILD,15,y,120,20,hPanel,NULL,hInst,NULL);
    hRfps = CreateWindowA("STATIC","0",WS_VISIBLE|WS_CHILD|SS_RIGHT,140,y,60,25,hPanel,NULL,hInst,NULL);
    SendMessage(hRfps,WM_SETFONT,(WPARAM)hf,1); y+=28;

    CreateWindowA("STATIC","",WS_VISIBLE|WS_CHILD|SS_ETCHEDHORZ,10,y,290,2,hPanel,NULL,hInst,NULL); y+=10;

    CreateWindowA("STATIC","Max Detect FPS:",WS_VISIBLE|WS_CHILD,15,y,120,20,hPanel,NULL,hInst,NULL);
    hFpsV = CreateWindowA("STATIC","300",WS_VISIBLE|WS_CHILD|SS_RIGHT,200,y,40,20,hPanel,NULL,hInst,NULL);
    SendMessage(hFpsV,WM_SETFONT,(WPARAM)hs,1); y+=22;
    hSFps = CreateWindowA(TRACKBAR_CLASSA,"",WS_VISIBLE|WS_CHILD|TBS_AUTOTICKS|TBS_TOOLTIPS,
        15,y,280,30,hPanel,NULL,hInst,NULL);
    SendMessage(hSFps,TBM_SETRANGE,TRUE,MAKELONG(5,300));
    SendMessage(hSFps,TBM_SETPOS,TRUE,300); y+=35;

    CreateWindowA("STATIC","Max Render FPS:",WS_VISIBLE|WS_CHILD,15,y,120,20,hPanel,NULL,hInst,NULL);
    hRfpsV = CreateWindowA("STATIC","144",WS_VISIBLE|WS_CHILD|SS_RIGHT,200,y,40,20,hPanel,NULL,hInst,NULL);
    SendMessage(hRfpsV,WM_SETFONT,(WPARAM)hs,1); y+=22;
    hSRfps = CreateWindowA(TRACKBAR_CLASSA,"",WS_VISIBLE|WS_CHILD|TBS_AUTOTICKS|TBS_TOOLTIPS,
        15,y,280,30,hPanel,NULL,hInst,NULL);
    SendMessage(hSRfps,TBM_SETRANGE,TRUE,MAKELONG(5,144));
    SendMessage(hSRfps,TBM_SETPOS,TRUE,60); y+=35;

    CreateWindowA("STATIC","FOV Radius:",WS_VISIBLE|WS_CHILD,15,y,120,20,hPanel,NULL,hInst,NULL);
    hFovV = CreateWindowA("STATIC","200",WS_VISIBLE|WS_CHILD|SS_RIGHT,200,y,40,20,hPanel,NULL,hInst,NULL);
    SendMessage(hFovV,WM_SETFONT,(WPARAM)hs,1); y+=22;
    hSFOV = CreateWindowA(TRACKBAR_CLASSA,"",WS_VISIBLE|WS_CHILD|TBS_AUTOTICKS,
        15,y,280,30,hPanel,NULL,hInst,NULL);
    SendMessage(hSFOV,TBM_SETRANGE,1,MAKELONG(50,640));
    SendMessage(hSFOV,TBM_SETPOS,1,200); y+=35;

    CreateWindowA("STATIC","Confidence:",WS_VISIBLE|WS_CHILD,15,y,120,20,hPanel,NULL,hInst,NULL);
    hConfV = CreateWindowA("STATIC","0.30",WS_VISIBLE|WS_CHILD|SS_RIGHT,200,y,40,20,hPanel,NULL,hInst,NULL);
    SendMessage(hConfV,WM_SETFONT,(WPARAM)hs,1); y+=22;
    hSConf = CreateWindowA(TRACKBAR_CLASSA,"",WS_VISIBLE|WS_CHILD|TBS_AUTOTICKS,
        15,y,280,30,hPanel,NULL,hInst,NULL);
    SendMessage(hSConf,TBM_SETRANGE,1,MAKELONG(5,95));
    SendMessage(hSConf,TBM_SETPOS,1,30); y+=35;

    CreateWindowA("STATIC","",WS_VISIBLE|WS_CHILD|SS_ETCHEDHORZ,10,y,290,2,hPanel,NULL,hInst,NULL); y+=10;

    CreateWindowA("STATIC","Detections:",WS_VISIBLE|WS_CHILD,15,y,120,20,hPanel,NULL,hInst,NULL);
    hDet = CreateWindowA("STATIC","0",WS_VISIBLE|WS_CHILD|SS_RIGHT,140,y,60,25,hPanel,NULL,hInst,NULL);
    SendMessage(hDet,WM_SETFONT,(WPARAM)hf,1); y+=30;

    hDebug = CreateWindowA("STATIC","--",WS_VISIBLE|WS_CHILD|SS_CENTER,
        10,y,290,20,hPanel,NULL,hInst,NULL);
    SendMessage(hDebug,WM_SETFONT,(WPARAM)hs,1); y+=24;

    CreateWindowA("STATIC","FOV Shape:",WS_VISIBLE|WS_CHILD,15,y,100,20,hPanel,NULL,hInst,NULL);
    hFovShape = CreateWindowA("BUTTON","Circle",WS_VISIBLE|WS_CHILD|BS_PUSHBUTTON,
        120,y,80,22,hPanel,NULL,hInst,NULL);
    SendMessage(hFovShape,WM_SETFONT,(WPARAM)hs,1); y+=28;

    CreateWindowA("STATIC","Track:",WS_VISIBLE|WS_CHILD,15,y,100,20,hPanel,NULL,hInst,NULL);
    hTrackLMB = CreateWindowA("BUTTON","LMB",WS_VISIBLE|WS_CHILD|BS_PUSHBUTTON,
        120,y,60,22,hPanel,NULL,hInst,NULL);
    SendMessage(hTrackLMB,WM_SETFONT,(WPARAM)hs,1);
    hTrackRMB = CreateWindowA("BUTTON","RMB",WS_VISIBLE|WS_CHILD|BS_PUSHBUTTON,
        185,y,60,22,hPanel,NULL,hInst,NULL);
    SendMessage(hTrackRMB,WM_SETFONT,(WPARAM)hs,1); y+=28;

    CreateWindowA("STATIC","Track Speed:",WS_VISIBLE|WS_CHILD,15,y,100,20,hPanel,NULL,hInst,NULL);
    hTrackSpeedV = CreateWindowA("STATIC","0.55",WS_VISIBLE|WS_CHILD|SS_RIGHT,200,y,40,20,hPanel,NULL,hInst,NULL);
    SendMessage(hTrackSpeedV,WM_SETFONT,(WPARAM)hs,1); y+=22;
    hSTrack = CreateWindowA(TRACKBAR_CLASSA,"",WS_VISIBLE|WS_CHILD|TBS_AUTOTICKS|TBS_TOOLTIPS,
        15,y,280,30,hPanel,NULL,hInst,NULL);
    SendMessage(hSTrack,TBM_SETRANGE,TRUE,MAKELONG(1,100));
    SendMessage(hSTrack,TBM_SETPOS,TRUE,50); y+=35;

    CreateWindowA("STATIC","Show Boxes:",WS_VISIBLE|WS_CHILD,15,y,100,20,hPanel,NULL,hInst,NULL);
    hShowBoxes = CreateWindowA("BUTTON","ON",WS_VISIBLE|WS_CHILD|BS_PUSHBUTTON,
        120,y,80,22,hPanel,NULL,hInst,NULL);
    SendMessage(hShowBoxes,WM_SETFONT,(WPARAM)hs,1); y+=28;

    RegisterHotKey(hPanel, 1, 0, VK_INSERT);
    RegisterHotKey(hPanel, 2, 0, VK_END);
    SetTimer(hPanel, 1, 500, NULL);

    ShowWindow(hPanel,SW_SHOW); UpdateWindow(hPanel);

    /* Auto-start inference thread */
    if(dbg) { fprintf(dbg, "Starting inference thread...\n"); fflush(dbg); }
    CreateThread(NULL,0,main_thread,NULL,0,NULL);
    CreateThread(NULL,0,track_thread,NULL,0,NULL);

    MSG msg = {};
    while(true) {
        while(PeekMessage(&msg,NULL,0,0,PM_REMOVE)) {
            if(msg.message==WM_QUIT) goto cleanup;
            if(msg.message==WM_HOTKEY) {
                if(msg.wParam==1) {
                    if(IsWindowVisible(hPanel)) ShowWindow(hPanel,SW_HIDE);
                    else ShowWindow(hPanel,SW_SHOW);
                }
                if(msg.wParam==2) goto cleanup;
            }
            TranslateMessage(&msg); DispatchMessage(&msg);
        }
        Sleep(1);
    }
cleanup:
    timeEndPeriod(1);
    g_running = false;
    if(g_sess) g_api->ReleaseSession(g_sess);
    if(g_mem) g_api->ReleaseMemoryInfo(g_mem);
    if(g_env) g_api->ReleaseEnv(g_env);
    cleanup_d3d();
    return 0;
}
