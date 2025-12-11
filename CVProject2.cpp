#define NOMINMAX

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <limits>
#include <cmath>
#include <filesystem>
#include <deque>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using std::cout;
using std::endl;
using std::string;

// ===================== CONFIG =====================

// 0 or negative => no cap, use all images found
static const int MAX_IMAGES = 0;

static const string DATASET_ROOT =
"enter your dataset";

static const int BRANCH_FACTOR = 8;   // k per node (max)
static const int MAX_DEPTH = 3;       // max depth of the tree
static const int K_EVAL = 10;         // precision@K
static const int MAX_EVAL_QUERIES = 100;

// UI sizes
static const int STATUS_WIN_W = 800;
static const int STATUS_WIN_H = 400;
static const int QUERY_WIN_W = 450;
static const int QUERY_WIN_H = 450;
static const int RESULTS_WIN_W = 1280;
static const int RESULTS_WIN_H = 720;

// ===================== Simple Status GUI =====================

std::deque<string> g_statusLines;

void pushStatus(const string& line)
{
    g_statusLines.push_back(line);
    if (g_statusLines.size() > 12) {
        g_statusLines.pop_front();
    }

    int width = STATUS_WIN_W;
    int height = STATUS_WIN_H;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(30, 30, 30));

    int y = 40;
    int lh = 22;

    cv::putText(canvas, "Vocabulary Tree Status",
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    for (const auto& s : g_statusLines) {
        cv::putText(canvas, s,
            cv::Point(10, y),
            cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        y += lh;
        if (y > height - 10) break;
    }

    cv::imshow("VT Status", canvas);
    cv::waitKey(1);
}

// ===================== Descriptor type =====================

enum class DescriptorType {
    SIFT,
    ORB,
    AKAZE
};

string descriptorTypeToString(DescriptorType t) {
    switch (t) {
    case DescriptorType::SIFT:  return "SIFT";
    case DescriptorType::ORB:   return "ORB";
    case DescriptorType::AKAZE: return "AKAZE";
    default:                    return "Unknown";
    }
}

// ===================== Feature extractor wrapper =====================

class FeatureExtractor {
public:
    explicit FeatureExtractor(DescriptorType type, int maxFeatures = 500)
        : type_(type)
    {
        if (type_ == DescriptorType::SIFT) {
            f2d_ = cv::SIFT::create(maxFeatures);
        }
        else if (type_ == DescriptorType::ORB) {
            f2d_ = cv::ORB::create(maxFeatures);
        }
        else if (type_ == DescriptorType::AKAZE) {
            f2d_ = cv::AKAZE::create();
        }
        else {
            throw std::runtime_error("Unsupported descriptor type");
        }
    }

    cv::Mat compute(const cv::Mat& img) const {
        cv::Mat gray;
        if (img.channels() == 3)
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else
            gray = img;

        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        f2d_->detectAndCompute(gray, cv::noArray(), kps, desc);

        if (desc.empty()) return cv::Mat();

        if (desc.type() != CV_32F)
            desc.convertTo(desc, CV_32F);

        return desc;
    }

    DescriptorType type() const { return type_; }

private:
    DescriptorType          type_;
    cv::Ptr<cv::Feature2D>  f2d_;
};

// ===================== Vocabulary Tree =====================

struct VocabNode {
    cv::Mat centroid;           // 1 x d, CV_32F
    std::vector<int> children;  // indices into nodes_
    bool isLeaf = false;
    int  wordId = -1;
};

class VocabularyTree {
public:
    VocabularyTree(int branchFactor = 8, int maxDepth = 3, int maxIters = 20)
        : B_(branchFactor), D_(maxDepth), iters_(maxIters)
    {
        VocabNode root;
        nodes_.push_back(root);
    }

    // ===== multi-level k-means tree build =====
    void build(const cv::Mat& descriptors) {
        if (descriptors.empty()) {
            throw std::runtime_error("Cannot build vocabulary from empty descriptor set.");
        }

        pushStatus("Building vocabulary tree (multi-level k-means)...");

        // Clear any previous tree and create a fresh root.
        nodes_.clear();

        VocabNode root;
        cv::reduce(descriptors, root.centroid, 0, cv::REDUCE_AVG);
        root.isLeaf = false;
        root.wordId = -1;
        nodes_.push_back(root);  // index 0 is root

        // BFS queue: each entry holds a node index, the subset of descriptors, and depth.
        struct BuildItem {
            int      nodeIdx;
            cv::Mat  data;
            int      depth;
        };

        std::deque<BuildItem> q;
        q.push_back({ 0, descriptors, 0 });

        int nextWordId = 0;
        const int MIN_SAMPLES_PER_NODE = B_ * 3; // heuristic uses instance B_

        while (!q.empty()) {
            BuildItem item = q.front();
            q.pop_front();

            int nodeIdx = item.nodeIdx;
            cv::Mat data = item.data;
            int depth = item.depth;

            if (nodeIdx < 0 || nodeIdx >= (int)nodes_.size()) {
                continue;
            }

            // Stopping criteria: max depth or too few samples → leaf
            if (depth >= D_ || data.rows < MIN_SAMPLES_PER_NODE) {
                nodes_[nodeIdx].isLeaf = true;
                nodes_[nodeIdx].wordId = nextWordId++;
                continue;
            }

            int effectiveK = std::min(B_, data.rows);
            if (effectiveK <= 1) {
                // Can't form at least 2 clusters → leaf
                nodes_[nodeIdx].isLeaf = true;
                nodes_[nodeIdx].wordId = nextWordId++;
                continue;
            }

            // k-means on this node's descriptors
            cv::Mat labels, centers;
            cv::TermCriteria crit(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                iters_, 1e-4);

            cv::kmeans(data, effectiveK, labels, crit, 1,
                cv::KMEANS_PP_CENTERS, centers);

            if (centers.empty() || labels.empty() || centers.rows != effectiveK) {
                // If k-means fails, make this node a leaf.
                nodes_[nodeIdx].isLeaf = true;
                nodes_[nodeIdx].wordId = nextWordId++;
                continue;
            }

            // Split data into child clusters
            std::vector<cv::Mat> childData(effectiveK);
            int nRows = std::min(data.rows, labels.rows);
            for (int i = 0; i < nRows; ++i) {
                int k = labels.at<int>(i, 0);
                if (k < 0 || k >= effectiveK) continue;
                childData[k].push_back(data.row(i));
            }

            // Create child nodes for non-empty clusters
            for (int k = 0; k < effectiveK; ++k) {
                if (childData[k].rows == 0) continue;

                VocabNode child;
                child.centroid = centers.row(k).clone();
                child.isLeaf = false;
                child.wordId = -1;

                int childIdx = static_cast<int>(nodes_.size());
                nodes_.push_back(child);
                nodes_[nodeIdx].children.push_back(childIdx);

                // Queue child for further splitting
                q.push_back({ childIdx, childData[k], depth + 1 });
            }

            // If no children ended up non-empty, fallback to leaf
            if (nodes_[nodeIdx].children.empty()) {
                nodes_[nodeIdx].isLeaf = true;
                nodes_[nodeIdx].wordId = nextWordId++;
            }
        }

        // Edge case: if we somehow never assigned any word IDs
        if (nextWordId == 0) {
            nodes_[0].isLeaf = true;
            nodes_[0].wordId = 0;
            nextWordId = 1;
        }

        numWords_ = nextWordId;

        pushStatus("Vocabulary built. Words = " + std::to_string(numWords_));
        cout << "Vocabulary built. Words: " << numWords_ << endl;
    }

    int numWords() const { return numWords_; }

    // Returns RAW term counts (no normalization here)
    cv::Mat quantize(const cv::Mat& descs) const {
        cv::Mat hist = cv::Mat::zeros(1, std::max(numWords_, 0), CV_32F);
        if (descs.empty() || numWords_ <= 0 || nodes_.empty()) return hist;

        for (int i = 0; i < descs.rows; ++i) {
            int w = traverseToLeaf(descs.row(i));
            if (w >= 0 && w < numWords_) {
                hist.at<float>(0, w) += 1.0f; // raw count
            }
        }

        return hist; // raw TF
    }

private:
    int B_;
    int D_;
    int iters_;
    std::vector<VocabNode> nodes_;
    int numWords_ = 0;

    int traverseToLeaf(const cv::Mat& descRow) const {
        if (nodes_.empty()) return -1;

        int nodeIdx = 0;
        while (true) {
            if (nodeIdx < 0 || nodeIdx >= (int)nodes_.size()) {
                return -1;
            }
            const auto& node = nodes_[nodeIdx];
            if (node.isLeaf || node.children.empty()) {
                return node.wordId;
            }

            int   bestChild = -1;
            float bestDist = std::numeric_limits<float>::max();
            for (int c : node.children) {
                if (c < 0 || c >= (int)nodes_.size()) continue;
                const cv::Mat& cent = nodes_[c].centroid;
                if (cent.empty() || cent.cols != descRow.cols) continue;
                cv::Mat diff = descRow - cent;
                float d = static_cast<float>(cv::norm(diff, cv::NORM_L2));
                if (d < bestDist) {
                    bestDist = d;
                    bestChild = c;
                }
            }
            if (bestChild < 0) {
                return node.wordId;
            }
            nodeIdx = bestChild;
        }
    }
};

// ===================== Index + Search =====================

struct IndexedImage {
    string path;
    string label;
    cv::Mat hist;   // TF or TF-IDF histogram (1 x W, CV_32F)
    cv::Mat descs;  // raw descriptors
};

class VTIndex {
public:
    VTIndex(const VocabularyTree& tree) : tree_(tree) {}

    void add(const string& path,
        const string& label,
        const cv::Mat& descs)
    {
        if (descs.empty()) return;

        // Get raw TF histogram from the tree
        cv::Mat h = tree_.quantize(descs);

        IndexedImage img;
        img.path = path;
        img.label = label;
        img.hist = h;
        img.descs = descs;
        imgs_.push_back(img);
    }

    // Compute IDF, apply TF-IDF weighting, then L2 normalize
    void finalize() {
        int N = static_cast<int>(imgs_.size());
        int W = tree_.numWords();

        if (N == 0 || W <= 0) {
            pushStatus("Index finalize: no images or no words, skipping TF-IDF.");
            return;
        }

        idf_ = cv::Mat::zeros(1, W, CV_32F);

        // Document frequency: count how many images have each word
        for (const auto& img : imgs_) {
            if (img.hist.empty() || img.hist.cols != W) continue;
            const float* hptr = img.hist.ptr<float>(0);
            for (int w = 0; w < W; ++w) {
                if (hptr[w] > 0.0f) {
                    idf_.at<float>(0, w) += 1.0f;
                }
            }
        }

        // Convert DF to IDF: idf = log((N + 1)/(df + 1)) + 1
        for (int w = 0; w < W; ++w) {
            float df = idf_.at<float>(0, w);
            float val = std::log((N + 1.0f) / (df + 1.0f)) + 1.0f;
            idf_.at<float>(0, w) = val;
        }

        // Apply TF-IDF and L2 normalize each image histogram
        for (auto& img : imgs_) {
            if (img.hist.empty() || img.hist.cols != W) {
                img.hist = cv::Mat::zeros(1, W, CV_32F);
                continue;
            }

            // TF-IDF: element-wise multiply (raw TF * IDF)
            img.hist = img.hist.mul(idf_);

            // L2 normalization for cosine similarity
            double nL2 = cv::norm(img.hist, cv::NORM_L2);
            if (nL2 > 0.0) {
                img.hist /= nL2;
            }
        }

        pushStatus("Index finalized (TF-IDF on raw TF + L2-normalized histograms).");
    }

    const std::vector<IndexedImage>& images() const { return imgs_; }

    const cv::Mat& idf() const { return idf_; }

private:
    const VocabularyTree& tree_;
    std::vector<IndexedImage>  imgs_;
    cv::Mat idf_;  // 1 x W, CV_32F
};

class VTSearcher {
public:
    VTSearcher(const VocabularyTree& tree,
        const VTIndex& index)
        : tree_(tree), index_(index)
    {
        const auto& imgs = index_.images();
        int N = static_cast<int>(imgs.size());
        int W = tree_.numWords();

        if (N <= 0 || W <= 0) {
            throw std::runtime_error("VTSearcher: invalid N or W.");
        }

        histMat_ = cv::Mat(N, W, CV_32F);
        for (int i = 0; i < N; ++i) {
            if (imgs[i].hist.empty() || imgs[i].hist.cols != W) {
                histMat_.row(i).setTo(0);
            }
            else {
                imgs[i].hist.copyTo(histMat_.row(i));
            }
        }

        // Copy IDF so we can TF-IDF weight queries the same way
        idf_ = index_.idf().clone();

        lastBestIndex_ = -1;
    }

    std::vector<std::pair<int, float>> search(const cv::Mat& qDescs) const {
        std::vector<std::pair<int, float>> res;

        if (qDescs.empty()) return res;

        // Quantize query descriptors into RAW TF histogram
        cv::Mat hist = tree_.quantize(qDescs);
        if (hist.empty() || hist.cols != histMat_.cols) {
            // Dimension mismatch or serious error: bail
            return res;
        }

        // TF-IDF weight the query if we have IDF
        if (!idf_.empty() && idf_.cols == hist.cols) {
            hist = hist.mul(idf_);
        }

        // L2-normalize query histogram
        double nL2 = cv::norm(hist, cv::NORM_L2);
        if (nL2 > 0.0) {
            hist /= nL2;
        }

        // Store for analysis later (for this query)
        lastQueryHist_ = hist.clone();

        int N = histMat_.rows;
        res.reserve(N);
        for (int i = 0; i < N; ++i) {
            float sim = hist.dot(histMat_.row(i)); // cosine similarity
            res.emplace_back(i, sim);
        }

        std::sort(res.begin(), res.end(),
            [](auto& a, auto& b) { return a.second > b.second; });

        if (!res.empty()) {
            lastBestIndex_ = res[0].first;
        }
        else {
            lastBestIndex_ = -1;
        }

        return res;
    }

    // Get the top-N word IDs contributing most to the similarity
    // between the last query and its best match.
    void getTopWordContributions(int topN,
        std::vector<int>& wordIds,
        std::vector<float>& contributions) const
    {
        wordIds.clear();
        contributions.clear();

        if (lastBestIndex_ < 0 ||
            lastBestIndex_ >= histMat_.rows ||
            lastQueryHist_.empty() ||
            lastQueryHist_.cols != histMat_.cols)
        {
            return;
        }

        int W = lastQueryHist_.cols;
        const float* qPtr = lastQueryHist_.ptr<float>(0);
        const float* dPtr = histMat_.ptr<float>(lastBestIndex_);

        std::vector<std::pair<int, float>> contribs;
        contribs.reserve(W);

        for (int w = 0; w < W; ++w) {
            float c = qPtr[w] * dPtr[w]; // contribution to cosine similarity
            if (c > 0.0f) {
                contribs.emplace_back(w, c);
            }
        }

        if (contribs.empty()) return;

        std::sort(contribs.begin(), contribs.end(),
            [](auto& a, auto& b) { return a.second > b.second; });

        int take = std::min(topN, (int)contribs.size());
        for (int i = 0; i < take; ++i) {
            wordIds.push_back(contribs[i].first);
            contributions.push_back(contribs[i].second);
        }
    }

private:
    const VocabularyTree& tree_;
    const VTIndex& index_;
    cv::Mat               histMat_;  // N x W
    cv::Mat               idf_;      // 1 x W

    // For analysis of the last search call
    mutable cv::Mat lastQueryHist_;  // 1 x W
    mutable int     lastBestIndex_;  // index in histMat_
};

// ===================== Dataset loading =====================

void loadDataset(const string& datasetRoot,
    std::vector<string>& paths,
    std::vector<string>& labels)
{
    namespace fs = std::filesystem;

    paths.clear();
    labels.clear();

    pushStatus("Loading dataset from: " + datasetRoot);

    fs::path root(datasetRoot);
    if (!fs::exists(root) || !fs::is_directory(root)) {
        cout << "[loadDataset] ERROR: root does not exist or is not a directory.\n";
        pushStatus("ERROR: dataset root not found.");
        return;
    }

    int classCount = 0;
    for (const auto& classEntry : fs::directory_iterator(root)) {
        if (!classEntry.is_directory()) continue;

        fs::path classPath = classEntry.path();
        string className = classPath.filename().string();
        int countInClass = 0;

        for (const auto& fileEntry : fs::directory_iterator(classPath)) {
            if (!fileEntry.is_regular_file()) continue;

            fs::path fpath = fileEntry.path();
            string ext = fpath.extension().string();

            std::transform(ext.begin(), ext.end(), ext.begin(),
                [](unsigned char c) { return (char)std::tolower(c); });

            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                paths.push_back(fpath.string());
                labels.push_back(className);
                countInClass++;
            }
        }

        if (countInClass > 0) {
            cout << "  Class: " << className
                << "  images: " << countInClass << endl;
            classCount++;
        }
    }

    cout << "[loadDataset] classes with images: " << classCount << endl;
    cout << "[loadDataset] total images found: " << paths.size() << endl;

    pushStatus("Dataset loaded: " + std::to_string(paths.size()) +
        " images, " + std::to_string(classCount) + " classes.");
}

// ===================== Evaluation metrics =====================

struct Metrics {
    double precisionAtK = 0.0;
    double recallAtK = 0.0;
    double mAP = 0.0;
};

Metrics evaluateDescriptor(
    DescriptorType descType,
    const VocabularyTree& tree,
    const VTIndex& index,
    const VTSearcher& searcher,
    int K,
    int maxQueries)
{
    (void)tree;

    const auto& imgs = index.images();
    int N = static_cast<int>(imgs.size());
    int numQueries = std::min(N, maxQueries);

    double sumP = 0.0;
    double sumR = 0.0;
    double sumAP = 0.0;
    int validQueries = 0;

    cout << "\n[Eval] Descriptor = " << descriptorTypeToString(descType)
        << ", K = " << K << ", queries = " << numQueries << endl;

    for (int qi = 0; qi < numQueries; ++qi) {
        if (qi < 0 || qi >= N) continue;
        const auto& qImg = imgs[qi];
        if (qImg.descs.empty()) continue;

        string qLabel = qImg.label;

        std::vector<int> relevant;
        for (int i = 0; i < N; ++i) {
            if (i == qi) continue;
            if (imgs[i].label == qLabel) {
                relevant.push_back(i);
            }
        }
        if (relevant.empty()) continue;

        auto ranking = searcher.search(qImg.descs);
        if (ranking.empty()) continue;

        int tp = 0;
        int retrieved = 0;
        double ap = 0.0;
        int hitsSoFar = 0;

        int rank = 0;
        for (const auto& p : ranking) {
            int idx = p.first;
            if (idx < 0 || idx >= N) continue;
            if (idx == qi) { rank++; continue; }

            bool isRel = (imgs[idx].label == qLabel);

            if (isRel) {
                hitsSoFar++;
                ap += static_cast<double>(hitsSoFar) / (rank + 1);
            }
            if (rank < K) {
                if (isRel) tp++;
                retrieved++;
            }

            rank++;
            if (rank >= std::max(K, (int)relevant.size() * 5)) {
                break;
            }
        }

        if (retrieved == 0) continue;

        double precK = static_cast<double>(tp) / (double)K;
        double recK = static_cast<double>(tp) / (double)relevant.size();
        double avgP = ap / (double)relevant.size();

        sumP += precK;
        sumR += recK;
        sumAP += avgP;
        validQueries++;
    }

    Metrics m;
    if (validQueries > 0) {
        m.precisionAtK = sumP / validQueries;
        m.recallAtK = sumR / validQueries;
        m.mAP = sumAP / validQueries;
    }

    cout << "[Eval] Valid queries: " << validQueries << endl;
    cout << "[Eval] Precision@" << K << " = " << m.precisionAtK << endl;
    cout << "[Eval] Recall@" << K << "    = " << m.recallAtK << endl;
    cout << "[Eval] mAP           = " << m.mAP << endl;

    return m;
}

// ===================== Pipeline result =====================

struct DescriptorPipelineResult {
    DescriptorType type;
    VocabularyTree tree;
    VTIndex        index;

    DescriptorPipelineResult(DescriptorType t,
        VocabularyTree&& tr,
        VTIndex&& idx)
        : type(t)
        , tree(std::move(tr))
        , index(std::move(idx))
    {
    }
};

// ===================== Helper: Pretty result panel =====================

// Create a pleasant canvas showing query (left) and top-2 matches stacked (right).
// extraInfo: text to show at the bottom (e.g., top words list).
cv::Mat createResultPanel(const cv::Mat& queryImg,
    const std::vector<cv::Mat>& matchImgs,
    const std::vector<string>& captions,
    const string& descriptorName,
    const string& extraInfo)
{
    int panelW = RESULTS_WIN_W;
    int panelH = RESULTS_WIN_H;
    cv::Mat canvas(panelH, panelW, CV_8UC3, cv::Scalar(15, 15, 18));

    // Header text
    string header = "Results \u2014 " + descriptorName;
    cv::putText(canvas, header,
        cv::Point(40, 50),
        cv::FONT_HERSHEY_SIMPLEX, 1.3,
        cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    cv::putText(canvas, "Left: Query   |   Right: Best & 2nd Best Matches",
        cv::Point(40, 85),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(180, 180, 180), 1, cv::LINE_AA);

    // Layout geometry
    int topMargin = 110;
    int bottomMargin = 70;
    int leftMargin = 40;
    int midMargin = 30;
    int rightMargin = 40;

    int contentHeight = panelH - topMargin - bottomMargin;
    int leftWidth = (int)(panelW * 0.40);             // query column
    int rightWidth = panelW - leftWidth - leftMargin - rightMargin - midMargin;

    // Helper: resize to fit in a box while preserving aspect
    auto fitIntoBox = [](const cv::Mat& src, int boxW, int boxH) {
        if (src.empty()) {
            return cv::Mat(boxH, boxW, CV_8UC3, cv::Scalar(40, 40, 40));
        }
        double scale = std::min((double)boxW / src.cols,
            (double)boxH / src.rows);
        if (scale <= 0.0) scale = 1.0;
        int newW = std::max(1, (int)std::round(src.cols * scale));
        int newH = std::max(1, (int)std::round(src.rows * scale));
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
        return dst;
        };

    // ---- Left: query image ----
    int queryBoxW = leftWidth;
    int queryBoxH = contentHeight;
    cv::Mat qImg = queryImg.empty()
        ? cv::Mat(queryBoxH, queryBoxW, CV_8UC3, cv::Scalar(50, 50, 50))
        : queryImg;

    cv::Mat qFit = fitIntoBox(qImg, queryBoxW, queryBoxH - 40);

    int qX = leftMargin + (queryBoxW - qFit.cols) / 2;
    int qY = topMargin + (queryBoxH - 40 - qFit.rows) / 2;

    qFit.copyTo(canvas(cv::Rect(qX, qY, qFit.cols, qFit.rows)));

    cv::putText(canvas, "Query",
        cv::Point(leftMargin + 5, topMargin + queryBoxH - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    // ---- Right: top-1 and top-2 stacked ----
    int rightX = leftMargin + leftWidth + midMargin;
    int rightY = topMargin;
    int boxH = contentHeight / 2 - 10;   // two boxes stacked
    int boxW = rightWidth;

    std::vector<cv::Mat> localMatches;
    localMatches.reserve(2);
    for (int i = 0; i < 2; ++i) {
        if (i < (int)matchImgs.size() && !matchImgs[i].empty()) {
            localMatches.push_back(matchImgs[i]);
        }
        else {
            localMatches.push_back(cv::Mat()); // will become gray
        }
    }

    std::vector<string> localCaps = captions;
    while (localCaps.size() < 2) {
        localCaps.push_back("No match");
    }

    for (int idx = 0; idx < 2; ++idx) {
        int boxTop = rightY + idx * (boxH + 10);
        cv::Mat mFit = fitIntoBox(localMatches[idx], boxW, boxH - 40);

        int mX = rightX + (boxW - mFit.cols) / 2;
        int mY = boxTop + (boxH - 40 - mFit.rows) / 2;
        mFit.copyTo(canvas(cv::Rect(mX, mY, mFit.cols, mFit.rows)));

        // Caption under each match
        cv::Scalar capColor = (idx == 0)
            ? cv::Scalar(0, 255, 0)
            : cv::Scalar(0, 200, 255);

        cv::putText(canvas, localCaps[idx],
            cv::Point(rightX + 5, boxTop + boxH - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.65,
            capColor, 1, cv::LINE_AA);
    }

    // ---- Extra info at bottom (top word IDs) ----
    if (!extraInfo.empty()) {
        cv::putText(canvas, extraInfo,
            cv::Point(40, panelH - 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }

    return canvas;
}

// ===================== Pipeline for one descriptor =====================

DescriptorPipelineResult runDescriptorPipeline(
    DescriptorType descType,
    const std::vector<string>& imgPaths,
    const std::vector<string>& labels)
{
    cout << "\n==============================\n";
    cout << "Running pipeline for descriptor: "
        << descriptorTypeToString(descType) << endl;
    cout << "Images: " << imgPaths.size() << endl;

    pushStatus("Pipeline: " + descriptorTypeToString(descType) +
        " - extracting features...");

    FeatureExtractor extractor(descType, 800);

    std::vector<cv::Mat> descsPerImg(imgPaths.size());
    std::vector<cv::Mat> allDescsVec;
    const int MIN_DESCS_PER_IMG = 20;

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < imgPaths.size(); ++i) {
        cv::Mat img = cv::imread(imgPaths[i], cv::IMREAD_COLOR);
        if (img.empty()) continue;

        cv::Mat d = extractor.compute(img);
        descsPerImg[i] = d;

        if (!d.empty() && d.rows >= MIN_DESCS_PER_IMG) {
            allDescsVec.push_back(d);
        }

        if (i % 200 == 0) {
            pushStatus(descriptorTypeToString(descType) +
                ": extracted " + std::to_string(i) + " / " +
                std::to_string(imgPaths.size()) + " images");
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double featTime = std::chrono::duration<double>(t1 - t0).count();
    cout << "Feature extraction time: " << featTime << " s" << endl;
    cout << "Images with enough features: " << allDescsVec.size() << endl;

    if (allDescsVec.empty()) {
        throw std::runtime_error("No descriptors extracted for " +
            descriptorTypeToString(descType));
    }

    cv::Mat allDescs;
    cv::vconcat(allDescsVec, allDescs);
    cout << "Total descriptor rows: " << allDescs.rows
        << ", dim: " << allDescs.cols << endl;

    pushStatus("Building vocab for " + descriptorTypeToString(descType) + "...");

    VocabularyTree tree(BRANCH_FACTOR, MAX_DEPTH);
    auto t2 = std::chrono::steady_clock::now();
    tree.build(allDescs);
    auto t3 = std::chrono::steady_clock::now();
    double treeTime = std::chrono::duration<double>(t3 - t2).count();
    cout << "Vocabulary built. Words: " << tree.numWords()
        << ", time: " << treeTime << " s" << endl;

    pushStatus("Indexing images for " + descriptorTypeToString(descType) + "...");

    VTIndex index(tree);
    int usedCount = 0;
    for (size_t i = 0; i < imgPaths.size(); ++i) {
        if (descsPerImg[i].empty()) continue;
        if (i >= labels.size()) continue;
        index.add(imgPaths[i], labels[i], descsPerImg[i]);
        usedCount++;
    }

    if (usedCount == 0) {
        throw std::runtime_error("Index has zero images for " +
            descriptorTypeToString(descType));
    }

    index.finalize();
    cout << "Indexed images: " << usedCount << endl;

    // Store tree and index in result (no searcher stored here)
    DescriptorPipelineResult result(descType,
        std::move(tree),
        std::move(index));

    // Local searcher for evaluation (safe: result.tree & result.index alive)
    VTSearcher evalSearcher(result.tree, result.index);

    pushStatus("Evaluating " + descriptorTypeToString(descType) + "...");

    evaluateDescriptor(descType,
        result.tree,
        result.index,
        evalSearcher,
        K_EVAL,
        MAX_EVAL_QUERIES);

    pushStatus("Pipeline done for " + descriptorTypeToString(descType) + ".");

    return result;
}

// ===================== MAIN =====================

int main()
{
    try {
        cv::ocl::setUseOpenCL(false);

        cv::namedWindow("VT Status", cv::WINDOW_NORMAL);
        cv::resizeWindow("VT Status", STATUS_WIN_W, STATUS_WIN_H);
        pushStatus("Starting program...");

        cout << "Dataset root = " << DATASET_ROOT << endl;

        // 1. Load dataset
        std::vector<string> imgPaths, labels;
        loadDataset(DATASET_ROOT, imgPaths, labels);

        if (imgPaths.empty()) {
            cout << "No images found. Check DATASET_ROOT and permissions.\n";
            pushStatus("No images found. Exiting.");
            cv::waitKey(0);
            return 0;
        }

        if (MAX_IMAGES > 0 && (int)imgPaths.size() > MAX_IMAGES) {
            imgPaths.resize(MAX_IMAGES);
            if (labels.size() > MAX_IMAGES) labels.resize(MAX_IMAGES);
        }

        cout << "Using " << imgPaths.size() << " images for this run.\n";
        pushStatus("Using " + std::to_string(imgPaths.size()) + " images.");

        // 2. Run pipelines for SIFT, ORB, AKAZE
        std::vector<DescriptorPipelineResult> pipelines;
        pipelines.reserve(3);

        std::vector<DescriptorType> types = {
            DescriptorType::SIFT,
            DescriptorType::ORB,
            DescriptorType::AKAZE
        };

        for (DescriptorType t : types) {
            try {
                pipelines.push_back(
                    runDescriptorPipeline(t, imgPaths, labels)
                );
            }
            catch (const std::exception& e) {
                std::cerr << "Error for descriptor "
                    << descriptorTypeToString(t)
                    << ": " << e.what() << endl;
                pushStatus("Error in " + descriptorTypeToString(t) +
                    ": " + e.what());
            }
        }

        if (pipelines.empty()) {
            cout << "No successful pipelines. Exiting.\n";
            pushStatus("No successful pipelines. Exiting.");
            cv::waitKey(0);
            return 0;
        }

        // Build searchers AFTER pipelines vector is complete
        std::vector<VTSearcher> searchers;
        searchers.reserve(pipelines.size());
        for (auto& pipe : pipelines) {
            searchers.emplace_back(pipe.tree, pipe.index);
        }

        pushStatus("Pipelines ready. Waiting for query images...");

        // Query window
        cv::namedWindow("Query Image", cv::WINDOW_NORMAL);
        cv::resizeWindow("Query Image", QUERY_WIN_W, QUERY_WIN_H);

        // Pre-create result windows for each descriptor
        for (auto& pipe : pipelines) {
            string winName = "Results - " + descriptorTypeToString(pipe.type);
            cv::namedWindow(winName, cv::WINDOW_NORMAL);
            cv::resizeWindow(winName, RESULTS_WIN_W, RESULTS_WIN_H);
        }

        // 3. Interactive query mode (console input)
        cout << "\n==============================\n";
        cout << "Interactive query mode.\n";
        cout << "Type a full image path and press Enter (or 'q' to quit).\n";

        while (true) {
            cout << "\nQuery image path> ";
            string qpath;
            if (!std::getline(std::cin, qpath)) {
                cout << "Input stream closed. Exiting.\n";
                pushStatus("Input stream closed. Exiting.");
                break;
            }

            if (qpath == "q" || qpath == "Q") {
                cout << "User requested quit.\n";
                pushStatus("User requested quit.");
                break;
            }
            if (qpath.empty()) {
                cout << "Empty input, try again.\n";
                continue;
            }

            cout << "\n[Query] Selected: " << qpath << endl;
            pushStatus("Query image: " + qpath);

            cv::Mat qImg = cv::imread(qpath, cv::IMREAD_COLOR);
            if (qImg.empty()) {
                cout << "[Query] ERROR: Could not read query image.\n";
                pushStatus("ERROR: Could not read query image.");
                continue;
            }

            cv::imshow("Query Image", qImg);
            cv::waitKey(1);

            for (size_t pi = 0; pi < pipelines.size(); ++pi) {
                auto& pipe = pipelines[pi];
                auto& searcher = searchers[pi];

                string descName = descriptorTypeToString(pipe.type);
                cout << "\n--- Descriptor: " << descName << " ---\n";

                pushStatus("[" + descName + "] Computing query descriptor...");

                FeatureExtractor extractor(pipe.type, 800);
                cv::Mat qDesc = extractor.compute(qImg);

                cout << "  Descriptor size: " << qDesc.rows
                    << " x " << qDesc.cols << endl;

                if (qDesc.empty()) {
                    cout << "  [!] No features found in query for this descriptor.\n";
                    pushStatus("[" + descName + "] No features found in query.");
                    continue;
                }

                pushStatus("[" + descName + "] Running search...");

                auto ranking = searcher.search(qDesc);
                if (ranking.empty()) {
                    cout << "  [!] No ranking results.\n";
                    pushStatus("[" + descName + "] Empty ranking.");
                    continue;
                }

                pushStatus("[" + descName + "] Search done. Showing top 5.");

                const auto& imgs = pipe.index.images();
                int topK = std::min(5, (int)ranking.size());
                for (int i = 0; i < topK; ++i) {
                    int   idx = ranking[i].first;
                    float sc = ranking[i].second;
                    if (idx < 0 || idx >= (int)imgs.size()) continue;
                    cout << "    #" << (i + 1)
                        << "  score = " << sc
                        << "  label = " << imgs[idx].label
                        << "  path = " << imgs[idx].path << endl;
                }

                // === Top word contributions for the BEST match ===
                std::vector<int>   topWordIds;
                std::vector<float> topWordContribs;
                searcher.getTopWordContributions(10, topWordIds, topWordContribs);

                if (!topWordIds.empty()) {
                    cout << "  Top contributing words (by cosine contribution) "
                        "for best match: ";
                    for (size_t i = 0; i < topWordIds.size(); ++i) {
                        cout << topWordIds[i];
                        if (i + 1 < topWordIds.size()) cout << " ";
                    }
                    cout << endl;
                }
                else {
                    cout << "  [Info] No positive word contributions found "
                        "for best match (possibly zero histogram).\n";
                }

                // === Fancy UI: show top-2 images in a panel ===
                std::vector<cv::Mat> matchImgs;
                std::vector<string>  captions;

                for (int i = 0; i < std::min(2, (int)ranking.size()); ++i) {
                    int idx = ranking[i].first;
                    float sc = ranking[i].second;
                    if (idx < 0 || idx >= (int)imgs.size()) continue;

                    cv::Mat mImg = cv::imread(imgs[idx].path, cv::IMREAD_COLOR);
                    if (mImg.empty()) {
                        mImg = cv::Mat(300, 300, CV_8UC3, cv::Scalar(40, 40, 40));
                    }
                    matchImgs.push_back(mImg);

                    char buf[512];
                    std::snprintf(buf, sizeof(buf),
                        "#%d  %s  (score = %.3f)",
                        i + 1, imgs[idx].label.c_str(), sc);
                    captions.emplace_back(buf);
                }

                // Format extra text for bottom: list of top words (shortened)
                string extraInfo;
                if (!topWordIds.empty()) {
                    size_t maxShow = std::min<size_t>(6, topWordIds.size());
                    extraInfo = "Top words in best match: ";
                    for (size_t i = 0; i < maxShow; ++i) {
                        extraInfo += std::to_string(topWordIds[i]);
                        if (i + 1 < maxShow) {
                            extraInfo += ", ";
                        }
                    }
                    if (topWordIds.size() > maxShow) {
                        extraInfo += ", ...";
                    }
                }

                cv::Mat panel = createResultPanel(qImg, matchImgs, captions, descName, extraInfo);
                string winName = "Results - " + descName;
                cv::imshow(winName, panel);
                cv::waitKey(1);
            }

            pushStatus("Query complete. Enter another path or 'q' to quit.");
        }

        cout << "Bye.\n";
        cv::waitKey(0);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        pushStatus(string("EXCEPTION: ") + e.what());
        cv::waitKey(0);
        return 1;
    }
}
