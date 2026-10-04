// layer_02.cpp
// A layer      = a function "tensor in -> tensor out" + the numbers it computes with.
// Weights      = those numbers. They are learned from data, nobody types them.
// A network    = a chain of layers.
// A model file = the chain's configuration + the value of every weight.
//
// This is the "inference side": it loads the model file that layer_02_pytorch.py
// trained and wrote, and runs it with ~20 lines of plain loops.
//
// Build:  g++ -std=c++17 -O2 layer_02.cpp -o layer_02
// Run:    ./layer_02 [path/to/line_model.txt]

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// CMake passes the absolute path; the fallback works when run from this folder.
#ifndef MODEL_FILE
#define MODEL_FILE "../line_model.txt"
#endif

// ---------------------------------------------------------------------------
// 1. Tensor: flat buffer + shape (see 01_tensor/Cpp/tensor_01.cpp for the full story)
// ---------------------------------------------------------------------------
struct Tensor {
    std::vector<float>  data;
    std::vector<size_t> shape;

    Tensor(std::vector<size_t> s, std::vector<float> d)
        : data(std::move(d)), shape(std::move(s)) {}

    size_t numel() const { return data.size(); }
};

void print_tensor(const std::string& name, const Tensor& t) {
    std::printf("  %-6s shape=(", name.c_str());
    for (size_t i = 0; i < t.shape.size(); ++i)
        std::printf("%zu%s", t.shape[i], i + 1 < t.shape.size() ? ", " : "");
    std::printf(")  [");
    for (float v : t.data) std::printf(" %+.3f", v);
    std::printf(" ]\n");
}

// ---------------------------------------------------------------------------
// 2. Layer: takes a tensor, computes, returns another tensor
// ---------------------------------------------------------------------------
struct Layer {
    std::string name;

    explicit Layer(std::string n) : name(std::move(n)) {}
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& x) const = 0;
    virtual size_t num_params() const { return 0; }
    virtual std::string type() const = 0;
};

// Fully-connected layer. Every output is connected to every input:
//   y[o] = sum_i weight[o][i] * x[i] + bias[o]
// Those "weight" and "bias" arrays ARE the weights of the layer.
struct Linear : Layer {
    size_t in, out;                // configuration
    std::vector<float> weight;     // out * in numbers, row o = weights of output o
    std::vector<float> bias;       // out numbers

    Linear(std::string n, size_t in_, size_t out_)
        : Layer(std::move(n)), in(in_), out(out_), weight(in_ * out_), bias(out_) {}

    Tensor forward(const Tensor& x) const override {
        if (x.numel() != in)
            throw std::invalid_argument(name + ": expected " + std::to_string(in) +
                                        " inputs, got " + std::to_string(x.numel()));
        Tensor y({out}, std::vector<float>(out));
        for (size_t o = 0; o < out; ++o) {
            float sum = bias[o];
            for (size_t i = 0; i < in; ++i)
                sum += weight[o * in + i] * x.data[i];
            y.data[o] = sum;
        }
        return y;
    }

    size_t num_params() const override { return weight.size() + bias.size(); }
    std::string type() const override { return "linear"; }
};

// ReLU: negatives become 0. A layer with NO weights: it only computes.
struct ReLU : Layer {
    using Layer::Layer;

    Tensor forward(const Tensor& x) const override {
        Tensor y = x;
        for (float& v : y.data)
            if (v < 0.0f) v = 0.0f;
        return y;
    }

    std::string type() const override { return "relu"; }
};

// ---------------------------------------------------------------------------
// 3. Network: a chain of layers. The output of one is the input of the next.
// ---------------------------------------------------------------------------
struct Model {
    std::vector<std::unique_ptr<Layer>> layers;

    Tensor forward(const Tensor& input, bool trace = false) const {
        Tensor x = input;
        if (trace) print_tensor("input", x);
        for (const auto& layer : layers) {
            x = layer->forward(x);
            if (trace) print_tensor(layer->name, x);
        }
        return x;
    }

    void print_summary() const {
        std::printf("  #  name   type    in -> out  params\n");
        size_t total = 0;
        for (size_t i = 0; i < layers.size(); ++i) {
            const Layer& l = *layers[i];
            std::string io = "same", note = "  (no weights: it only computes)";
            if (auto* fc = dynamic_cast<const Linear*>(&l)) {
                io = std::to_string(fc->in) + " -> " + std::to_string(fc->out);
                note = "  (" + std::to_string(fc->out) + "*" + std::to_string(fc->in) +
                       " weights + " + std::to_string(fc->out) + " biases)";
            }
            std::printf("  %zu  %-6s %-7s %-10s %zu%s\n", i + 1, l.name.c_str(),
                        l.type().c_str(), io.c_str(), l.num_params(), note.c_str());
            total += l.num_params();
        }
        std::printf("  total params = %zu  (= %zu bytes as float32)\n", total,
                    total * sizeof(float));
    }
};

// ---------------------------------------------------------------------------
// 4. Two ways to get a Model: random weights, or weights from a model file
// ---------------------------------------------------------------------------
// Same configuration as the file, but the weights are random noise.
Model make_untrained_model(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    auto fc1 = std::make_unique<Linear>("fc1", 9, 6);
    auto fc2 = std::make_unique<Linear>("fc2", 6, 2);
    for (Linear* fc : {fc1.get(), fc2.get()}) {
        for (float& w : fc->weight) w = dist(rng);
        for (float& b : fc->bias)   b = dist(rng);
    }

    Model m;
    m.layers.push_back(std::move(fc1));
    m.layers.push_back(std::make_unique<ReLU>("relu1"));
    m.layers.push_back(std::move(fc2));
    return m;
}

// The file says which layers exist, in what order, and the value of every weight.
// Nothing about the network is hard-coded here: change the file, get another network.
Model load_model(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open model file: " + path);

    // Drop '#' comments; what is left is one whitespace-separated token stream
    std::stringstream ss;
    for (std::string line; std::getline(file, line);)
        ss << line.substr(0, line.find('#')) << '\n';

    auto expect = [&](const std::string& word) {
        std::string tok;
        if (!(ss >> tok) || tok != word)
            throw std::runtime_error("model file: expected '" + word + "', got '" + tok + "'");
    };
    auto read_floats = [&](std::vector<float>& v) {
        for (float& f : v)
            if (!(ss >> f)) throw std::runtime_error("model file: not enough numbers");
    };

    size_t n_layers = 0;
    expect("layers");
    if (!(ss >> n_layers)) throw std::runtime_error("model file: bad layer count");

    Model m;
    for (size_t i = 0; i < n_layers; ++i) {
        std::string type, name;
        if (!(ss >> type >> name)) throw std::runtime_error("model file: missing layer");
        if (type == "linear") {
            size_t in = 0, out = 0;
            if (!(ss >> in >> out)) throw std::runtime_error("model file: bad linear size");
            auto fc = std::make_unique<Linear>(name, in, out);
            expect("weight");
            read_floats(fc->weight);
            expect("bias");
            read_floats(fc->bias);
            m.layers.push_back(std::move(fc));
        } else if (type == "relu") {
            m.layers.push_back(std::make_unique<ReLU>(name));
        } else {
            throw std::runtime_error("model file: unknown layer type '" + type + "'");
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// 5. Test data: six 3x3 images, flattened to 9 numbers
// ---------------------------------------------------------------------------
const char* const CLASSES[] = {"vertical", "horizontal"};

struct Sample {
    Tensor image;
    int label;
};

std::vector<Sample> line_images() {
    std::vector<Sample> samples;
    for (size_t k = 0; k < 3; ++k) {
        std::vector<float> v(9, 0.0f), h(9, 0.0f);
        for (size_t j = 0; j < 3; ++j) {
            v[j * 3 + k] = 1.0f;   // column k is lit
            h[k * 3 + j] = 1.0f;   // row k is lit
        }
        samples.push_back({Tensor({9}, v), 0});
        samples.push_back({Tensor({9}, h), 1});
    }
    return samples;
}

void evaluate(const Model& model, const std::vector<Sample>& samples) {
    size_t correct = 0;
    for (const Sample& s : samples) {
        Tensor scores = model.forward(s.image);
        int pred = scores.data[1] > scores.data[0];   // the bigger score wins
        correct += pred == s.label;

        std::array<std::string, 3> rows;
        for (size_t i = 0; i < 9; ++i) rows[i / 3] += s.image.data[i] > 0.5f ? '#' : '.';

        std::string verdict = pred == s.label
            ? "OK" : std::string("WRONG (it is ") + CLASSES[s.label] + ")";
        std::printf("  %s   scores: vertical=%+.3f  horizontal=%+.3f\n", rows[0].c_str(),
                    scores.data[0], scores.data[1]);
        std::printf("  %s   -> %s  %s\n", rows[1].c_str(), CLASSES[pred], verdict.c_str());
        std::printf("  %s\n", rows[2].c_str());
    }
    std::printf("  correct: %zu/%zu\n\n", correct, samples.size());
}

// ---------------------------------------------------------------------------
// 6. Demo
// ---------------------------------------------------------------------------
int main(int argc, char** argv) try {
    const std::string model_path = argc > 1 ? argv[1] : MODEL_FILE;

    // (a) ONE layer, weights typed by hand so the arithmetic is visible
    std::printf("(a) One layer: Linear(3 -> 2), hand-typed weights\n");
    Linear layer("demo", 3, 2);
    layer.weight = {1.0f, 0.0f, -1.0f,     // output 0
                    0.5f, 0.5f,  0.5f};    // output 1
    layer.bias   = {0.0f, 1.0f};
    Tensor x({3}, {2.0f, 3.0f, 4.0f});
    Tensor y = layer.forward(x);
    print_tensor("x", x);
    std::printf("  y[0] = 1.0*2 + 0.0*3 + -1.0*4 + 0.0 = %+.1f\n", y.data[0]);
    std::printf("  y[1] = 0.5*2 + 0.5*3 +  0.5*4 + 1.0 = %+.1f\n", y.data[1]);
    print_tensor("y", y);
    print_tensor("relu", ReLU("relu").forward(y));
    std::printf("\n");

    const std::vector<Sample> samples = line_images();

    // (b) The right layers but random weights: the network knows nothing
    std::printf("(b) Untrained network (random weights)\n");
    Model untrained = make_untrained_model(1);
    untrained.print_summary();
    evaluate(untrained, samples);

    // (c) Load the model file: configuration + weights learned by PyTorch
    std::printf("(c) Network loaded from %s\n", model_path.c_str());
    Model model = load_model(model_path);
    model.print_summary();
    evaluate(model, samples);

    // (d) Follow one image through the chain: every layer is tensor in, tensor out
    std::printf("(d) One image through the chain, layer by layer\n");
    Tensor out = model.forward(samples[0].image, /*trace=*/true);
    std::printf("  -> %s\n\n", CLASSES[out.data[1] > out.data[0]]);

    // (b) and (c) ran the exact same code. Only the 74 numbers differ.
    // ResNet-18 is the same idea with 11,689,512 numbers (~46.8 MB as float32).
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
