// Fast O(n) rolling z-score: each point is compared with the previous `w` points.
// Auto-built by Python: g++ -O2 -shared -fPIC -o libfast_analysis.so fast_analysis.cpp
#include <cmath>
extern "C" {
#ifdef _WIN32
__declspec(dllexport)
#endif
void rolling_zscore(const double* x, int n, int w, double* out) {
    double s = 0.0, s2 = 0.0;
    for (int i = 0; i < n; i++) {
        int cnt = i < w ? i : w;
        if (cnt < 8) {
            out[i] = 0.0;
        } else {
            double m = s / cnt;
            double var = s2 / cnt - m * m;
            double sd = std::sqrt(var > 0 ? var : 0.0);
            if (sd < 1e-3) sd = 1e-3;
            out[i] = (x[i] - m) / sd;
        }
        s += x[i]; s2 += x[i] * x[i];
        if (i >= w) { s -= x[i - w]; s2 -= x[i - w] * x[i - w]; }
    }
}
}
