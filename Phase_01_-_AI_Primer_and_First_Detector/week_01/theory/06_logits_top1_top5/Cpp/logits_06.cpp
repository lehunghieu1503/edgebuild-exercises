// logits_06.cpp
// Reading the output of a classification network.
//
//   logits : the raw numbers the last layer returns, one per class.
//            The largest one is the class the network picks.
//   top-1  : fraction of images where the highest-scoring class is the correct one
//   top-5  : fraction of images where the correct class is among the 5 highest scores
//
// The logits here are made up: this file is about READING them, not producing them.
//
// Build:  g++ -std=c++17 -O2 logits_06.cpp -o logits_06
// Run:    ./logits_06

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

// Tiny deterministic random generator, so C++ and Python produce the same numbers.
struct Lcg {
    uint32_t state;
    float next() {   // uniform in [-0.5, 0.5)
        state = state * 1664525u + 1013904223u;
        return (state >> 8) / 16777216.0f - 0.5f;
    }
};

const char* const CLASSES[10] = {"airplane", "automobile", "bird", "cat", "deer",
                                 "dog", "frog", "horse", "ship", "truck"};

// ---------------------------------------------------------------------------
// 1. Reading one logits vector
// ---------------------------------------------------------------------------
// The prediction: index of the largest logit
size_t argmax(const std::vector<float>& logits) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

// Indices of the k largest logits, best first
std::vector<size_t> top_k(const std::vector<float>& logits, size_t k) {
    std::vector<size_t> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](size_t a, size_t b) { return logits[a] > logits[b]; });
    idx.resize(k);
    return idx;
}

// Logits are not probabilities (they can be negative, they do not add up to 1).
// Softmax converts them. It keeps the order, so the winner stays the winner.
std::vector<float> softmax(const std::vector<float>& logits) {
    float max = *std::max_element(logits.begin(), logits.end());
    std::vector<float> p(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < p.size(); ++i) sum += p[i] = std::exp(logits[i] - max);
    for (float& v : p) v /= sum;
    return p;
}

// ---------------------------------------------------------------------------
// 2. Accuracy over many images
// ---------------------------------------------------------------------------
// Rank of the correct class: 1 = it has the highest score, 2 = one class beats it, ...
// No sorting needed: just count how many classes score higher.
size_t rank_of(const std::vector<float>& logits, size_t label) {
    size_t higher = 0;
    for (float v : logits) higher += v > logits[label];
    return higher + 1;
}

// top-k accuracy = fraction of images whose correct class has rank <= k
float top_k_accuracy(const std::vector<std::vector<float>>& batch,
                     const std::vector<size_t>& labels, size_t k) {
    size_t hits = 0;
    for (size_t i = 0; i < batch.size(); ++i) hits += rank_of(batch[i], labels[i]) <= k;
    return (float)hits / batch.size();
}

// ---------------------------------------------------------------------------
// 3. Demo
// ---------------------------------------------------------------------------
int main() {
    // (a) One image, 10 classes -> 10 logits
    std::printf("(a) One image -> 10 logits (one per class)\n");
    std::vector<float> logits = {-1.2f, 0.3f, 2.1f, 5.7f, 1.0f, 4.9f, 0.2f, 1.8f, -0.5f, -2.0f};
    std::printf("  logits = [");
    for (float v : logits) std::printf(" %+.1f", v);
    std::printf(" ]\n");
    std::printf("  argmax = %zu -> the network says \"%s\"\n", argmax(logits),
                CLASSES[argmax(logits)]);

    std::vector<float> probs = softmax(logits);
    std::printf("  top 5:  rank  index  class       logit  probability\n");
    std::vector<size_t> best = top_k(logits, 5);
    for (size_t r = 0; r < best.size(); ++r)
        std::printf("          %4zu  %5zu  %-10s  %+5.1f  %.3f\n", r + 1, best[r],
                    CLASSES[best[r]], logits[best[r]], probs[best[r]]);
    std::printf("\n");

    // (b) Eight images with known labels -> top-1 and top-5 accuracy
    std::printf("(b) 8 labeled images -> top-1 and top-5\n");
    const std::vector<size_t> labels = {3, 5, 0, 8, 1, 9, 2, 6};
    // How strongly each made-up image "looks like" its true class
    const std::vector<float> boost = {3.0f, 3.0f, 2.0f, 1.2f, 1.2f, 0.6f, 0.6f, -1.5f};
    Lcg rng{11};
    std::vector<std::vector<float>> batch;
    for (size_t i = 0; i < labels.size(); ++i) {
        std::vector<float> row(10);
        for (float& v : row) v = 4.0f * rng.next();
        row[labels[i]] += boost[i];
        batch.push_back(row);
    }

    std::printf("  image  label       predicted   rank of label  top-1  top-5\n");
    for (size_t i = 0; i < batch.size(); ++i) {
        size_t rank = rank_of(batch[i], labels[i]);
        std::printf("  %5zu  %-10s  %-10s  %13zu  %-5s  %s\n", i, CLASSES[labels[i]],
                    CLASSES[argmax(batch[i])], rank, rank <= 1 ? "hit" : "-",
                    rank <= 5 ? "hit" : "-");
    }
    std::printf("  top-1 accuracy = %.3f\n", top_k_accuracy(batch, labels, 1));
    std::printf("  top-5 accuracy = %.3f   (always >= top-1)\n\n", top_k_accuracy(batch, labels, 5));

    // (c) ImageNet-sized output: 1000 classes -> 1000 logits per image
    std::printf("(c) A 1000-class network returns 1000 logits per image\n");
    std::vector<float> big(1000);
    for (float& v : big) v = 20.0f * rng.next();
    std::printf("  number of logits = %zu\n", big.size());
    std::printf("  top 5 indices =");
    for (size_t i : top_k(big, 5)) std::printf(" %zu (%+.2f)", i, big[i]);
    std::printf("\n  a table index -> class name turns the winning index into a word\n");
    return 0;
}
