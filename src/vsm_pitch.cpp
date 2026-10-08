#include "openv6/vsm_pitch.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <optional>
#include <set>

#include "openv6/util.hpp"

namespace openv6::vsm_pitch {

namespace {

const float Up = std::bit_cast<float>(0x3f949b93u);
const float Down = std::bit_cast<float>(0x3f5c8000u);
constexpr double VsmRate = 44100.0 / 256;
constexpr double Big = 1e10;

int vsm_frame(double seconds) { return static_cast<int>(std::floor(round_away(seconds * 44100) / 256 + 0.5)); }

int clamp_lower_first(int x, int lo, int hi) {
    if (x < lo) x = lo;
    return hi >= x ? x : hi;
}

std::vector<KeyPoint> key_points(std::span<const float> curve) {
    int n = static_cast<int>(curve.size()), m = n >> 2;
    std::set<int> idx = {0, n - 1};
    if (m == 0) {
        idx.insert(static_cast<int>(n * 0.5f + 0.5f));
    } else {
        float step = (n - 1.0f) / (2 * m);
        for (int i = 1; i < 2 * m; i++) idx.insert(static_cast<int>(i * step + 0.5f));
    }
    std::vector<KeyPoint> points;
    for (int i : idx) points.push_back({i, curve[i]});
    return points;
}

}

std::vector<float> resample(std::span<const float> x, float ratio) {
    int n = static_cast<int>(x.size());
    int m = static_cast<int>(std::floor(n / ratio + 0.5f));
    std::vector<float> output(m);
    for (int i = 0; i < m; i++) {
        float pos = i * ratio;
        int u = static_cast<int>(std::floor(pos));
        if (u == 0 || u >= n - 3) {
            output[i] = x[std::min(u, n - 1)];
            continue;
        }
        float p0 = x[u], p1 = x[u + 1], p2 = x[u + 2], pm = x[u - 1];
        float t = pos - u;
        if (p0 == Gap) {
            if (p1 == Gap || !(t > 0.5f)) {
                output[i] = Gap;
                continue;
            }
            p0 = p1;
        } else if (p1 == Gap) {
            if (t > 0.5f) {
                output[i] = Gap;
                continue;
            }
            p1 = p0;
        }
        if (pm == Gap) pm = p0;
        if (p2 == Gap) p2 = p1;
        float a = (p2 - p0) * 0.5f, b = (p1 - pm) * 0.5f;
        output[i] = (a - (p1 + p1) + b + p0 + p0) * t * t * t + (p1 * 3.0f - a - (b + b) - p0 * 3.0f) * t * t + b * t + p0;
    }
    return output;
}

std::vector<float> constrained_spline(std::span<const KeyPoint> points) {
    int n = static_cast<int>(points.size());
    std::vector<int> x(n);
    std::vector<double> y(n);
    for (int i = 0; i < n; i++) {
        x[i] = points[i].x;
        y[i] = std::clamp((points[i].v + 6900.0) / 13200.0, 0.0, 1.0);
    }
    std::vector<float> output;
    auto emit = [&](double value) { output.push_back(std::clamp(static_cast<float>(value) * 13200.0f - 6900.0f, -6900.0f, 6300.0f)); };
    if (n == 1) {
        emit(y[0]);
        return output;
    }
    if (n == 2) {
        int steps = x[1] - x[0];
        for (int k = 0; k < steps; k++) emit(k * ((y[1] - y[0]) / steps) + y[0]);
        emit(y[1]);
        return output;
    }
    auto dx_of = [&](int a, int b) { return x[a] != x[b] ? static_cast<double>(x[a] - x[b]) : Big; };
    auto slope = [&](int j) {
        if (j == 0 || j == n - 1) return Big;
        double dy1 = y[j + 1] - y[j], dy0 = y[j] - y[j - 1];
        if (dy1 == 0 || dy0 == 0) return 0.0;
        double dx1 = dx_of(j + 1, j), dx0 = dx_of(j, j - 1);
        if (dx0 / dy0 + dx1 / dy1 == 0 || dy0 * dy1 < 0) return 0.0;
        return 2.0 / (dx0 / dy0 + dx1 / dy1);
    };
    for (int i = 1; i < n; i++) {
        double s0 = slope(i - 1), s1 = slope(i);
        if (i == 1) {
            s0 = (y[1] - y[0]) * 1.5 / dx_of(1, 0) - s1 * 0.5;
        } else if (i == n - 1) {
            s1 = (y[i] - y[i - 1]) * 1.5 / dx_of(i, i - 1) - s0 * 0.5;
        }
        double dx = dx_of(i, i - 1), dy = y[i] - y[i - 1];
        double f0 = dy * 6.0 / (dx * dx) - 2.0 * (s0 + s0 + s1) / dx;
        double f1 = 2.0 * (s1 + s1 + s0) / dx - dy * 6.0 / (dx * dx);
        double d = (f1 - f0) * 0.16666666666666666 / dx;
        double xa = x[i - 1], xb = x[i];
        double c = (f0 * xb - xa * f1) * 0.5 / dx;
        double b = (dy - (xb * xb - xa * xa) * c - (xb * xb * xb - xa * xa * xa) * d) / dx;
        double a = y[i - 1] - xa * b - xa * xa * c - xa * xa * xa * d;
        for (int k = 0; k < x[i] - x[i - 1]; k++) {
            double t = x[i - 1] + k;
            emit(t * t * c + t * b + a + t * t * t * d);
        }
    }
    output.push_back(points[n - 1].v);
    return output;
}

std::vector<float> note_pitch(std::span<const float> pitch, std::span<const float> residual, int midi, const AiNoteExpression& e) {
    int n = static_cast<int>(pitch.size());
    std::vector<float> base(n);
    for (int i = 0; i < n; i++) base[i] = pitch[i] - residual[i];
    auto keys = key_points(base);
    auto smooth = constrained_spline(keys);
    float note = static_cast<float>(midi - 69) * 100.f;
    float fine = (e.pitch_fine - 0.5f) * 200.f;
    float scale_center = e.pitch_scaling_center + e.pitch_scaling_center;
    float scale_origin = e.pitch_scaling_origin + e.pitch_scaling_origin;
    int half = n >> 1;
    std::map<int, float> warped;
    for (size_t k = 0; k < keys.size(); k++) {
        auto [frame, value] = keys[k];
        if (k == 0 || k + 1 == keys.size()) {
            warped[frame] = value;
            continue;
        }
        int s;
        float weight, drift;
        if (frame < half) {
            float x = static_cast<float>(frame) * e.pitch_transition_start;
            s = clamp_lower_first(static_cast<int>(x + x), 1, half - 1);
            weight = std::sqrt(1.f - static_cast<float>(s) / static_cast<float>(half));
            drift = e.pitch_drift_start;
        } else {
            float x = static_cast<float>(n - frame) * e.pitch_transition_end;
            s = clamp_lower_first(static_cast<int>(static_cast<float>(n) - (x + x)), half, n - 2);
            weight = std::sqrt(static_cast<float>(s - half) / static_cast<float>(half));
            drift = e.pitch_drift_end;
        }
        float t = weight * -1.f * 2.f * (drift - 0.5f);
        float v = lerp(value, note, t);
        v = lerp(note, v, scale_center);
        v = lerp(note, v, scale_origin);
        v += fine;
        if (frame < half || !warped.contains(s)) warped[s] = v;
    }
    std::vector<KeyPoint> moved_keys;
    for (auto [frame, value] : warped) moved_keys.push_back({frame, value});
    auto moved = constrained_spline(moved_keys);
    for (int i = 0; i < n; i++) moved[i] = moved[i] + (smooth[i] - base[i]) * scale_origin;
    return moved;
}

namespace {

int sign(float x) { return x < 0 ? -1 : x > 0 ? 1 : 0; }

std::vector<float> moving_average5(std::span<const float> x) {
    int n = static_cast<int>(x.size());
    std::vector<float> out(n);
    if (n < 5) {
        float sum = 0;
        for (float v : x) sum += v;
        std::fill(out.begin(), out.end(), sum / static_cast<float>(n));
        return out;
    }
    float acc = 0;
    for (int i = 0; i < n; i++) {
        if (i < 3 || n <= i + 2) {
            acc = 0;
            int count = 0;
            for (int j = i - 2; j <= i + 2; j++) {
                if (j < 0 || j >= n) continue;
                acc += x[j];
                count++;
            }
            out[i] = acc / static_cast<float>(count);
        } else {
            acc = acc - x[i - 3] + x[i + 2];
            out[i] = acc / 5.f;
        }
    }
    return out;
}

struct VibratoTemplate {
    int period;
    float amplitude;
    std::vector<float> shape;
};

std::optional<VibratoTemplate> residual_cycle(std::span<const float> residual, int a, int b) {
    int period = b - a;
    if (period < 9 || period > 144) return std::nullopt;
    std::vector<float> shape;
    float total_sin = 0, total_residual = 0;
    for (int k = a; k <= b; k++) {
        float s = static_cast<float>(std::sin((k - a) * 6.283185307179586 / period));
        total_sin += std::fabs(s);
        total_residual += std::fabs(residual[k]);
        shape.push_back(s);
    }
    float amplitude = total_residual / total_sin;
    if (!(amplitude <= 600 && 6 <= amplitude)) return std::nullopt;
    int outside = 0;
    for (size_t k = 1; k + 1 < shape.size(); k++) {
        shape[k] = residual[a + k] / amplitude - shape[k];
        if (!(std::fabs(shape[k]) < 1)) outside++;
    }
    if (!(static_cast<float>(outside) <= static_cast<float>(period) * 0.5f)) return std::nullopt;
    return VibratoTemplate{period, amplitude, std::move(shape)};
}

std::vector<VibratoTemplate> residual_templates(std::span<const float> residual) {
    auto smooth = moving_average5(residual);
    std::vector<int> crossings;
    int previous = sign(smooth[0]);
    for (size_t i = 1; i < smooth.size(); i++) {
        int current = sign(smooth[i]);
        if (previous <= 0 && current > 0) crossings.push_back(static_cast<int>(i));
        previous = current;
    }
    std::vector<VibratoTemplate> templates;
    for (size_t i = 0; i + 1 < crossings.size(); i++)
        if (auto cycle = residual_cycle(residual, crossings[i], crossings[i + 1])) templates.push_back(std::move(*cycle));
    return templates;
}

}

void add_vibrato(std::span<float> pitch, std::span<const float> residual, int depth, int rate, bool ai, int start, const AiNoteExpression& e) {
    int n = static_cast<int>(pitch.size());
    if (depth <= 0 || n - 2 <= start) return;
    std::vector<VibratoTemplate> templates;
    if (ai) templates = residual_templates(residual);
    int mean_period = 0x24;
    if (templates.empty()) {
        templates.push_back({0x24, 60.f, std::vector<float>(0x25, 0.f)});
    } else {
        int sum = 0;
        for (const auto& t : templates) sum += t.period;
        mean_period = sum / static_cast<int>(templates.size());
    }
    float factor = 1.f;
    if (rate < 64) {
        factor = static_cast<float>(rate) * 0.015625f + 0.5f;
    } else if (rate > 64) {
        factor = static_cast<float>(rate) / 127.f;
        factor += factor;
    }
    int period = std::max(2, static_cast<int>(static_cast<float>(mean_period) / factor));
    int cycles = (n + period - 1) / period;
    int count = static_cast<int>(templates.size());
    std::vector<int> order;
    for (int j = 0; j < cycles; j++)
        if (j * period >= start) order.push_back(count * j / cycles);
    if (order.empty()) order.push_back(count - 1);
    size_t next = 0;
    int current = order[next++];
    int half = (n - start) / 2;
    int mid = start + half;
    float fade_step = 0, fade = 1;
    int phase = 0;
    float lead_scale = (e.vibrato_leading_depth + e.vibrato_leading_depth) - 1.f;
    float follow_scale = (e.vibrato_following_depth + e.vibrato_following_depth) - 1.f;
    for (int f = start; f < n; f++) {
        if (phase == period) {
            if (next < order.size()) current = order[next++];
            phase = 0;
            if (next == order.size()) fade_step = 1.f / static_cast<float>(n - f);
        }
        const auto& t = templates[current];
        float position = static_cast<float>(phase) / static_cast<float>(period) * static_cast<float>(t.period);
        int k = static_cast<int>(position);
        float shaped = lerp(t.shape[k], t.shape[k + 1], position - static_cast<float>(k));
        float gain = static_cast<float>(depth) * 0.015625f;
        float u = f > mid ? static_cast<float>(f - mid) / static_cast<float>(half) * follow_scale + 1.f
                          : (1.f - static_cast<float>(f - mid + half) / static_cast<float>(half)) * lead_scale + 1.f;
        gain *= static_cast<float>(static_cast<double>(u) * u);
        if (fade_step > 0) {
            gain *= static_cast<float>(static_cast<double>(fade) * fade);
            fade -= fade_step;
        }
        float wave = std::sin(static_cast<float>(phase * 6.283185307179586 / period));
        pitch[f] += (wave + shaped) * t.amplitude * gain;
        phase++;
    }
}

std::vector<PitchNote> pitch_notes(std::span<const V6Note> notes, std::span<const V6Phoneme> timed) {
    std::vector<PitchNote> result;
    for (size_t i = 0; i < notes.size(); i++) {
        const auto& note = notes[i];
        if (note.rest()) continue;
        const V6Phoneme *first = nullptr, *last = nullptr;
        for (const auto& p : timed) {
            if (p.note != static_cast<int>(i)) continue;
            if (!first) first = &p;
            last = &p;
        }
        int midi = static_cast<int>(round_even((*note.pitch - note.pitch_offset) / 100.f)) + 69;
        result.push_back({static_cast<int>(round_away(first->start * VsmRate)), static_cast<int>(round_away(last->end * VsmRate)), midi,
                          note.pitch_offset, note.expression, note.vibrato_depth, note.vibrato_rate, note.ai_vibrato,
                          vsm_frame(note.vibrato_start)});
    }
    return result;
}

DirectPitchCurve direct_pitch(const V6PitchEdit& edit) {
    auto frame_of = vsm_frame;
    auto time_of = [&](int64_t tick) { return edit.origin_seconds + tick * edit.tick_seconds; };
    std::map<int64_t, double> anchors;
    for (size_t i = 0; i < edit.ticks.size(); i++) {
        double snapped = frame_of(time_of(edit.ticks[i])) * 256 / 44100.0;
        anchors[static_cast<int64_t>(std::floor((snapped - edit.origin_seconds) / edit.tick_seconds + 0.5))] = edit.midi_cents[i];
    }
    std::vector<int64_t> at;
    std::vector<double> av;
    for (auto [tick, cents] : anchors) {
        at.push_back(tick);
        av.push_back(cents);
    }
    std::map<int, float> frames;
    auto set = [&](int64_t tick, float cents) { frames[frame_of(time_of(tick))] = cents; };
    double kept = av[0];
    set(at[0], static_cast<float>(kept));
    for (size_t seg = 0; seg + 1 < at.size(); seg++) {
        for (int64_t t = at[seg] + 1; t <= at[seg + 1]; t++) {
            double v = av[seg] + (av[seg + 1] - av[seg]) * (t - at[seg]) / (at[seg + 1] - at[seg]);
            if (std::fabs(v - kept) < 1) continue;
            kept = v;
            set(t, static_cast<float>(v));
        }
    }
    set(at.back() + 1, std::numeric_limits<float>::quiet_NaN());
    int first = frames.begin()->first;
    std::vector<float> cents(frames.rbegin()->first - first + 1);
    for (size_t f = 0; f < cents.size(); f++) {
        auto it = frames.find(first + static_cast<int>(f));
        cents[f] = it != frames.end() ? it->second : cents[f - 1];
    }
    return {first, std::move(cents)};
}

std::vector<float> f0_curve(std::span<const float> pitch, std::span<const float> residual, std::span<const PitchNote> notes,
                            int frames, const V6PitchEdit* edit) {
    std::vector<float> p(frames, Gap), r(frames, Gap);
    std::copy(pitch.begin(), pitch.end(), p.begin());
    std::copy(residual.begin(), residual.end(), r.begin());
    auto up = resample(p, Up);
    auto up_residual = resample(r, Up);
    for (const auto& note : notes)
        for (int i = note.lo; i < std::min(note.hi, static_cast<int>(up.size())); i++) up[i] -= note.pitch_offset;
    auto output = up;
    for (const auto& note : notes) {
        int hi = std::min(note.hi, static_cast<int>(up.size()) - 1);
        if (hi - note.lo + 1 > 1) {
            auto span = [&](const std::vector<float>& v) { return std::span<const float>(v).subspan(note.lo, hi - note.lo + 1); };
            auto curve = note_pitch(span(up), span(up_residual), note.midi, note.expression);
            add_vibrato(curve, span(up_residual), note.vibrato_depth, note.vibrato_rate, note.ai_vibrato, std::max(note.vibrato_start, note.lo) - note.lo,
                        note.expression);
            std::copy(curve.begin(), curve.end(), output.begin() + note.lo);
        }
    }
    if (edit) {
        auto [first, cents] = direct_pitch(*edit);
        for (size_t i = 0; i < cents.size() && first + i < output.size(); i++)
            if (!std::isnan(cents[i])) output[first + i] = cents[i] - 6900.0f;
    }
    auto f0 = resample(output, Down);
    std::vector<float> result(frames, Gap);
    std::copy_n(f0.begin(), std::min<size_t>(frames, f0.size()), result.begin());
    return result;
}

}
