#pragma once

class MatchBox
{
public:
    MatchBox() {
        memset(this, 0, sizeof(MatchBox));
    };
    MatchBox(int X1, int Y1, int X2, int Y2, float sc,float psc, float Gray = 0,size_t l = 0) {
        x1 = X1; y1 = Y1; x2 = X2; y2 = Y2; score = sc; pixsc = psc; gray = Gray; labelnum = l;
    }
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    float pixsc;
    float gray;
    size_t labelnum;
    float area() { return getWidth() * getHeight(); };
    cv::Point getp1() { return cv::Point(x1, y1); };
    cv::Point getp2() { return cv::Point(x2, y2); };
    cv::Rect getRECT() { return cv::Rect(x1, y1, getWidth(), getHeight()); };
    float getWidth() { return (x2 - x1); };
    float getHeight() { return (y2 - y1); };
};

struct imginfo
{
    std::string name;
    cv::Mat img;
    double precision;
};

enum class RBGCHANNEL
{
    red = 0,
    green=1,
    blue=2
};
class PictureMatch
{
public:
	cv::Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2);
    MatchBox getmatchbox(cv::Mat& targetImage, cv::Mat& templateImage);
    std::vector<MatchBox>  getmatchboxes(cv::Mat& targetImage, std::vector<cv::Mat>& templateImages);
    std::vector<MatchBox> getmatchboxs(cv::Mat& targetImage, cv::Mat& templateImage,double inthreshold = 0.9);
    void MAT2WBC(cv::Mat& image, cv::Mat& edges,bool boostborder = false);//增加识别的成功率
    float pixelmatching(cv::Mat& image1, cv::Mat& image2, int Failover = 10);
    cv::Mat dealwithblue(cv::Mat& image);
    cv::Mat dealcolor(cv::Mat& image,cv::Scalar color,int diff = 0);
    float BrightnessEqualizationContrast(cv::Mat& img1, cv::Mat& img2);
    float SingleChannelPixelComparison(cv::Mat& img1,cv::Mat&img2, RBGCHANNEL channel,int gap=20);
    float BGR2GRAYCMP(cv::Mat& img1, cv::Mat& img2, double SplitValue);
    double picpixSimilarity(cv::Mat& img1, cv::Mat& img2);
    MatchBox picpixMatch(cv::Mat& imgsrc, cv::Mat& imgtel);
    std::vector<MatchBox> picpixMatch(cv::Mat& imgsrc, std::vector<cv::Mat>& imgtels);
    float binwper(cv::Mat& img);
    float redper(cv::Mat& img);
    float blueper(cv::Mat& img);
    std::vector<cv::Rect> gettextrects(cv::Mat& img2,uint minarea = 98, uint minhight = 14);
    std::vector<cv::Mat> gettextimgs(cv::Mat& img2, uint minarea = 98, uint minhight = 14);
    std::vector<cv::Mat> gettextimgs2Vampix(cv::Mat& img2, uint minarea = 98, uint minhight = 14);
    cv::Mat img_BGR2GRAY3C(cv::Mat& img);
    cv::Mat img_BGR2GRAY3C(cv::Mat& img,double t1, double t2,int type);
    double calculate_brightness(cv::Mat& image);
    double calculate_blue_ratio(const cv::Mat& image);
private:
    double pixelSimilarity(int p1R, int p1G, int p1B, int p2R, int p2G, int p2B);
    double pixelSimilarity(cv::Vec3b& p1, cv::Vec3b& p2);
    double getgraylightestthresh(cv::Mat& image, double thresh);
    double getgraythreshLDR(cv::Mat& image, uchar threshdif);//LDR算法取最佳阈值
    double getgrayprp(cv::Mat& image);
    template<typename T>
    T square(T num);
};

