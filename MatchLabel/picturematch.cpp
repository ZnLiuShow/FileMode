#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "PictureMatch.h"
#include <math.h>
#include <numeric>

cv::Scalar PictureMatch::getMSSIM(const cv::Mat& i1, const cv::Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2 = I2.mul(I2);        // I2^2
    cv::Mat I1_2 = I1.mul(I1);        // I1^2
    cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    cv::Scalar mssim = mean(ssim_map); // mssim = average of ssim map
    return mssim;
}

double PictureMatch::getgrayprp(cv::Mat& image) {
    size_t i = 0;
    for (size_t y = 0; y < image.rows; y++)
    {
        for (size_t x = 0; x < image.cols; x++)
        {
            if (image.ptr<cv::Vec3b>(y, x)->val[0] == image.ptr<cv::Vec3b>(y, x)->val[1] && image.ptr<cv::Vec3b>(y, x)->val[0] == image.ptr<cv::Vec3b>(y, x)->val[2])
                i++;     
        }
    }
    return (double)i / (double)image.size().area();
}

MatchBox PictureMatch::getmatchbox(cv::Mat& targetImage, cv::Mat& templateImage) {
    if (targetImage.empty() || templateImage.empty())
        return MatchBox();
    if (targetImage.size().height < templateImage.size().height || targetImage.size().width < templateImage.size().width)
        return MatchBox();
    // ����ģ��ƥ��
    cv::Mat result; //����ƥ����ͼƬ
    cv::matchTemplate(targetImage, templateImage, result, cv::TM_CCOEFF_NORMED);

    // Ѱ�����ƥ��λ��
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);  

    cv::Mat img0 = templateImage;
    cv::Mat img1 = targetImage(cv::Rect(maxLoc.x, maxLoc.y, templateImage.cols, templateImage.rows));
    return   MatchBox(maxLoc.x, maxLoc.y, maxLoc.x + templateImage.cols, maxLoc.y + templateImage.rows, (float)maxVal,picpixSimilarity(img0, img1), getgrayprp(img1));
}

std::vector<MatchBox>  PictureMatch::getmatchboxes(cv::Mat& targetImage, std::vector<cv::Mat>& templateImages) {
    std::vector<MatchBox> r;
    for (auto& templateImage : templateImages)
    {
        // ����ģ��ƥ��
        cv::Mat result; //����ƥ����ͼƬ
        cv::matchTemplate(targetImage, templateImage, result, cv::TM_CCOEFF_NORMED);

        // Ѱ�����ƥ��λ��
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        auto i = &templateImage - &templateImages[0];
        cv::Mat img0 = templateImage;
        cv::Mat img1 = targetImage(cv::Rect(maxLoc.x, maxLoc.y, templateImage.cols, templateImage.rows));
        r.push_back(MatchBox(maxLoc.x, maxLoc.y, maxLoc.x + templateImage.cols, maxLoc.y + templateImage.rows, (float)maxVal, picpixSimilarity(img0, img1), getgrayprp(img1),i));
    }
    return r;
}

std::vector<MatchBox> PictureMatch::getmatchboxs(cv::Mat& image, cv::Mat& templateImage,double inthreshold) {
    // ׼��������
    cv::Mat result;
    result.create(image.cols - templateImage.cols + 1, image.rows - templateImage.rows + 1, CV_32F);

    // ƥ��ģ��
    cv::matchTemplate(image, templateImage, result, cv::TM_CCOEFF_NORMED);

    // �������ƥ��λ��
    std::vector<MatchBox> locations;
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    // �ദƥ��
    double threshold = inthreshold * maxVal; // ��ֵϵ�������Ը�����Ҫ����
    while (true) {
        // �ҵ�������ֵ��λ��
        if (maxVal <= threshold) {
            break;
        }
        cv::Mat img0 = templateImage.clone();
        cv::Mat img1 = image(cv::Rect(maxLoc.x, maxLoc.y, templateImage.cols, templateImage.rows));
        locations.push_back(MatchBox(maxLoc.x, maxLoc.y, maxLoc.x + templateImage.cols, maxLoc.y + templateImage.rows, (float)maxVal, picpixSimilarity(img0, img1), getgrayprp(img1)));

        // �ƶ��������ڣ�����ƥ��
        cv::floodFill(result, maxLoc, cv::Scalar(0));
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    }
    return locations;
}

void PictureMatch::MAT2WBC(cv::Mat& image, cv::Mat& edges,bool boostborder) {
    // ת���ɻҶ�ͼ
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);
    if (boostborder)
    {
        // ��ѡ��ʹ����̬ѧ������ǿ�߽�
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(edges, edges, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 1);
    }
}


float PictureMatch::pixelmatching(cv::Mat& image1, cv::Mat& image2, int Failover) {
    int all = image1.rows * image1.cols;
    int cout = 0;
    // ת��Ϊ�Ҷ�ͼ
    cv::Mat gray1;
    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    // ת��Ϊ�Ҷ�ͼ
    cv::Mat gray2;
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    // ����ͼƬ�е�ÿ������
    for (int y = 0; y < image1.rows; ++y) {
        for (int x = 0; x < image1.cols; ++x) {
            // ��ȡ����ͼƬ��(x, y)λ�õ�����ֵ
            uchar pixel1 = gray1.at<uchar>(y, x);
            uchar pixel2 = gray2.at<uchar>(y, x);
            // �Ա�����ֵ������ֻ��BGR�����ļ򵥶Ա�
            if (abs((int)pixel1- (int)pixel2) <= Failover) {
                cout++;
            }
        }
    }
    return   (float)cout / (float)all;
}

cv::Mat PictureMatch::dealwithblue(cv::Mat& image) {
    // ����һ����ԭͼ��ͬ��С����Ĥ������0���
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    // ����ͼƬ�е�ÿ������
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // ��ȡ(x, y)λ�õ�����ֵ
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            // ������ز�������ɫ�ģ�����Ĥ�еĶ�Ӧλ����Ϊ255
            if ((uchar)color[2] < (uchar)color[0] || (uchar)color[1]  > (uchar)color[2]) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }
    // ʹ����Ĥ��ȡ��ɫ������
    cv::Mat bluePixels;
    image.copyTo(bluePixels, mask);
    return bluePixels;
}

cv::Mat PictureMatch::dealcolor(cv::Mat& image, cv::Scalar color, int diff) {
    // ����һ����ԭͼ��ͬ��С����Ĥ������0���
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    // ����ͼƬ�е�ÿ������
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // ��ȡ(x, y)λ�õ�����ֵ
            uchar* p = image.ptr<uchar>(y, x);
            // ������ز�������ɫ�ģ�����Ĥ�еĶ�Ӧλ����Ϊ255
            if (!(abs((int)p[0] - (int)color.val[0]) <= diff && abs((int)p[1] - (int)color.val[1]) <= diff && abs((int)p[2] - (int)color.val[2]) <= diff)) {
                mask.at<uchar>(y, x) = 255;
            }
        }
        std::cout << std::endl;
    }
    cv::Mat Pixels;
    image.copyTo(Pixels, mask);
    return Pixels;
}

float PictureMatch::BrightnessEqualizationContrast(cv::Mat& img1, cv::Mat& img2) {
    // ��������ͼƬ��ֱ��ͼ���⻯ �����ǵ�ͨ��
    cv::Mat histogramEqualizedImg1, histogramEqualizedImg2;
    cv::equalizeHist(img1, histogramEqualizedImg1);
    cv::equalizeHist(img2, histogramEqualizedImg2);
    cv::imshow("histogramEqualizedImg1", histogramEqualizedImg1);
    cv::imshow("histogramEqualizedImg2", histogramEqualizedImg2);
    cv::waitKey(0);
    int all = img1.rows * img1.cols;
    int cout = 0;
    std::vector<double> data;
    for (int y = 0; y < img1.rows; ++y) {
        for (int x = 0; x < img1.cols; ++x) {
            // ��ȡ����ͼƬ��(x, y)λ�õ�����ֵ
            uchar pixel1 = histogramEqualizedImg1.at<uchar>(y, x);
            uchar pixel2 = histogramEqualizedImg2.at<uchar>(y, x);
            // �Ա�����ֵ������ֻ��BGR�����ļ򵥶Ա�
            if (abs((int)pixel1 - (int)pixel2)<=20) {
                cout++;
            }      
        }
    }
    return  (float)cout/(float)all;
}

float PictureMatch::SingleChannelPixelComparison(cv::Mat& img1, cv::Mat& img2, RBGCHANNEL channel, int gap) {
    int cout = 0;

    // ����ͼƬ�е�ÿ������
    for (int y = 0; y < img1.rows; ++y) {
        for (int x = 0; x < img2.cols; ++x) {
            // ��ȡ����ͼƬ��(x, y)λ�õ�����ֵ
            cv::Vec3b pixel1 = img1.at<cv::Vec3b>(y, x);
            cv::Vec3b pixel2 = img2.at<cv::Vec3b>(y, x);
            if (abs(pixel1[(size_t)channel] - pixel2[(size_t)channel]) <= 20) {
                cout++;
            }
        }
    }
    return (float)cout / (float)(img1.rows * img1.cols);
}

float binaryimagepixelcmp(cv::Mat& image1, cv::Mat& image2) {
    int all = image1.rows * image1.cols;
    int cout = 0;

    // ����ͼƬ�е�ÿ������
    for (int y = 0; y < image1.rows; ++y) {
        for (int x = 0; x < image1.cols; ++x) {
            // ��ȡ����ͼƬ��(x, y)λ�õ�����ֵ
            uchar pixel1 = image1.at<uchar>(y, x);
            uchar pixel2 = image2.at<uchar>(y, x);
            // �Ա�����ֵ������ֻ��BGR�����ļ򵥶Ա�
            if (pixel1 == pixel2) {
                cout++;
            }
        }
    }
    return   (float)cout / (float)all;
}

float PictureMatch::BGR2GRAYCMP(cv::Mat& img1, cv::Mat& img2,double SplitValue) {
    cv::Mat gray1;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::Mat gray2;
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    double thresh1 = getgraylightestthresh(gray1, SplitValue);
    double thresh2 = getgraylightestthresh(gray2, SplitValue);
    cv::Mat edge1;
    cv::threshold(gray1, edge1, thresh1, 255, cv::THRESH_BINARY); 
    cv::Mat edge2;
    cv::threshold(gray2, edge2, thresh2, 255, cv::THRESH_BINARY);
    //cv::imshow("gray1", gray1);
    //cv::imshow("gray2", gray2);
    //cv::imshow("edge1", edge1);
    //cv::imshow("edge2", edge2);
    //cv::waitKey(0);
    return  binaryimagepixelcmp(edge1, edge2);
}

double PictureMatch::getgraylightestthresh(cv::Mat& image, double thresh) {
    int t = 0;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // ��ȡ(i, j)λ�õ�����ֵ
            uchar pixelValue = image.at<uchar>(y, x);
            t += (int)pixelValue;
        }
    }
    float autothresh = t / (image.rows * image.cols) * thresh;
    if (autothresh > 255)
        autothresh = 255;
    return autothresh;
}

double PictureMatch::getgraythreshLDR(cv::Mat& image, uchar threshdif) {
    int once = 0;
    uchar t_thresh = 0;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // ��ȡ(i, j)λ�õ�����ֵ
            uchar pixelValue = image.at<uchar>(y, x);
            if (once == 0)
            {
                once = 1;
                t_thresh = pixelValue;
            }
            if (abs((int)pixelValue - t_thresh)<=10)
            {

            }
            else
            {

            }
        }
    }
}
double PictureMatch::picpixSimilarity(cv::Mat& img1, cv::Mat& img2) {
    if (img1.size() != img2.size())
        return 0;
    double s = 0.0;
    for (size_t y = 0; y < img1.rows; y++)
    {
        for (size_t x = 0; x < img1.cols; x++)
        {
            //std::cout << pixelSimilarity(img1.at<cv::Vec3b>(y, x), img2.at<cv::Vec3b>(y, x)) << "  ";
            s += pixelSimilarity(*img1.ptr<cv::Vec3b>(y, x), *img2.ptr<cv::Vec3b>(y, x));  
        }
        //std::cout << std::endl;
    }

    return s / (double)img1.size().area();
}

MatchBox PictureMatch::picpixMatch(cv::Mat& imgsrc, cv::Mat& imgtel) {
    if (imgsrc.rows < imgtel.rows || imgsrc.cols < imgtel.cols)
        return MatchBox();
    size_t width = imgtel.cols;
    size_t hight = imgtel.rows;
    double ps = 0;
    MatchBox result= MatchBox();
    for (size_t y = 0; y <= imgsrc.rows - hight; y++)
    {
     
        for (size_t x = 0; x <= imgsrc.cols - width; x++)
        {
            cv::Mat roi = imgsrc(cv::Rect(x, y, width, hight));
            double pcs = picpixSimilarity(roi, imgtel); 
            if (ps < pcs)
            {
                result = MatchBox(x,y,x + width,y+ hight,0, pcs);
                ps = pcs;
            }
        }
    }
    return result;
}

std::vector<MatchBox> PictureMatch::picpixMatch(cv::Mat& imgsrc, std::vector<cv::Mat>& imgtels) {
    for (auto& imgtel : imgtels)
    {
        if (imgsrc.rows < imgtel.rows || imgsrc.cols < imgtel.cols)
            return std::vector<MatchBox>();
    }
    std::vector<double> ps(imgtels.size(),0);
    std::vector<MatchBox> result(imgtels.size(), MatchBox());
    for (size_t y = 0; y < imgsrc.rows; y++){

        for (size_t x = 0; x < imgsrc.cols; x++){
            for (size_t i = 0; i < imgtels.size(); i++)
            {
                size_t width = imgtels[i].cols;
                size_t hight = imgtels[i].rows;
                if (x + width > imgsrc.cols || y + hight > imgsrc.rows)//Խ��
                    continue;
                cv::Mat roi = imgsrc(cv::Rect(x, y, width, hight));
                double pcs = picpixSimilarity(roi, imgtels[i]);
                if (ps[i] < pcs)
                {
                    result[i] = MatchBox(x, y, x + width, y + hight, 0, pcs,i);
                    ps[i] = pcs;
                }
            }
        }
    }
    return result;
}
double calculateVariance(int data[], int length) {
    double mean = 0.0, variance = 0.0;
    int i;
    // Calculate mean
    for (i = 0; i < length; i++) {
        mean += data[i];
    }
    mean /= length;

    // Calculate variance
    for (i = 0; i < length; i++) {
        variance += (data[i] - mean) * (data[i] - mean);
    }
    variance /= length;
    return variance;
}

double PictureMatch::pixelSimilarity(int p1R, int p1G, int p1B, int p2R, int p2G, int p2B) {
    //������������֮���ŷ�Ͼ�����Ϊ���ƶȵĶ���
    int spR = p1R - p2R;
    int spG = p1G - p2G;
    int spB = p1B - p2B;
    double distance_p = square(spR) + square(spG) + square(spB);
    // ����ֵ��Χ����Ϊ0��255
    double maxDistance_p = 3 * 255 * 255;
    // ���ؾ�����������ܾ���Ľ����Ϊ���ƶ�[0,1]
    return 1 - sqrt(distance_p / maxDistance_p);
}

double PictureMatch::pixelSimilarity(cv::Vec3b& p1, cv::Vec3b& p2) {
    return pixelSimilarity((int)p1[0], (int)p1[1], (int)p1[2], (int)p2[0], (int)p2[1], (int)p2[2]);
}

template<typename T>
T PictureMatch::square(T num) {
    return num * num;
}

float  PictureMatch::binwper(cv::Mat& img) {
    int cout = 0;
    // ����ͼƬ�е�ÿ������
    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3b* rowPtr = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img.cols; ++x) {
            // ��ȡ����ͼƬ��(x, y)λ�õ�����ֵ
            cv::Vec3b* pixelPtr = &rowPtr[x];
            if (pixelPtr->val[0] > 10) {
                cout++;
            }
        }
    }
    return (float)cout / (float)img.size().area();
}

float PictureMatch::redper(cv::Mat& img) {
    int cout = 0;
    // ����ͼƬ�е�ÿ������
    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3b* rowPtr = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img.cols; ++x) {
            // ��ȡ����ͼƬ��(x, y)λ�õ�����ֵ
            cv::Vec3b* pixelPtr = &rowPtr[x];
            if ((float)(pixelPtr->val[2] - pixelPtr->val[1]) / (float)pixelPtr->val[2] > 0.4 && (float)(pixelPtr->val[2] - pixelPtr->val[0]) / (float)pixelPtr->val[2] > 0.4) {
                cout++;
            }
        }
    }
    return (float)cout / (float)img.size().area();
}
float PictureMatch::blueper(cv::Mat& img) {
    int cout = 0;
    // ����ͼƬ�е�ÿ������
    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3b* rowPtr = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img.cols; ++x) {
            // ��ȡ����ͼƬ��(x, y)λ�õ�����ֵ
            cv::Vec3b* pixelPtr = &rowPtr[x];
            if ((float)(pixelPtr->val[0] - pixelPtr->val[1])/(float)pixelPtr->val[0] >0.4 && (float)(pixelPtr->val[0] - pixelPtr->val[2]) / (float)pixelPtr->val[0] > 0.4) {
                cout++;
            }
        }
    }
    return (float)cout / (float)img.size().area();
}

std::vector<cv::Rect> PictureMatch::gettextrects(cv::Mat& img2, uint minarea, uint minhight) {
    // ��ȡͼƬ������ͼƬ�Ǻ�ɫ���ذ�Χ������
    cv::Mat img;
    cv::cvtColor(img2, img, cv::COLOR_BGR2GRAY);
    // ��ֵ���������Ǻ�ɫ�����ر�Ϊ��ɫ
    cv::Mat thresh;
    cv::threshold(img, thresh, 1, 255, cv::THRESH_BINARY_INV);

    // Ѱ������
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> r;
    for (size_t i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) < minarea)
            continue;
        // ���������ı߽����
        cv::Rect rect = cv::boundingRect(contours[i]);
        if (rect.height >= (int)minhight)
            r.push_back(rect);
    }
    return r;
}

std::vector<cv::Mat> PictureMatch::gettextimgs(cv::Mat& img2,uint minarea, uint minhight) {
    // ��ȡͼƬ������ͼƬ�Ǻ�ɫ���ذ�Χ������
    cv::Mat img;
    cv::cvtColor(img2, img, cv::COLOR_BGR2GRAY);
    // ��ֵ���������Ǻ�ɫ�����ر�Ϊ��ɫ
    cv::Mat thresh;
    cv::threshold(img, thresh, 1, 255, cv::THRESH_BINARY_INV);

    // Ѱ������
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Mat> r;
    for (size_t i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) < minarea)
            continue;
        // ���������ı߽����
        cv::Rect rect = cv::boundingRect(contours[i]);
        if (rect.height >= (int)minhight)
            r.push_back(img2(rect).clone());
    }
    return r;
}

std::vector<cv::Mat>  PictureMatch::gettextimgs2Vampix(cv::Mat& img2, uint minarea , uint minhight ) {
    // ��ȡͼƬ������ͼƬ�Ǻ�ɫ���ذ�Χ������
    cv::Mat img;
    cv::cvtColor(img2, img, cv::COLOR_BGR2GRAY);
    // ��ֵ���������Ǻ�ɫ�����ر�Ϊ��ɫ
    cv::Mat thresh;
    cv::threshold(img, thresh,74, 255, cv::THRESH_BINARY_INV);
    //cv::imshow("person", thresh);

    // Ѱ������
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Mat> r;
    for (size_t i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) < minarea)
            continue;
        // ���������ı߽����
        cv::Rect rect = cv::boundingRect(contours[i]);
        if (rect.height >= (int)minhight) {
            cv::Mat grayImage = img(rect);
            cv::Mat binaryImage = cv::Mat::zeros(grayImage.size(), grayImage.type());
            cv::threshold(grayImage, binaryImage, 250, 255, cv::THRESH_BINARY);
            cv::Mat threeChannelImage;
            std::vector<cv::Mat> channels = { binaryImage,  binaryImage,  binaryImage };
            cv::merge(channels, threeChannelImage);
            r.push_back(threeChannelImage);
        }       
    }

    //for (size_t i = 0; i < r.size(); i++)
    //{
    //    cv::imshow(std::to_string(i), r[i]);
    //}
    return r;
}

cv::Mat PictureMatch::img_BGR2GRAY3C(cv::Mat& img) {
    cv::Mat grayImage;
    // ����ɫͼƬת��Ϊ�Ҷ�ͼƬ
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat binaryImage = cv::Mat::zeros(grayImage.size(), grayImage.type());
    cv::threshold(grayImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat threeChannelImage;
    std::vector<cv::Mat> channels = { binaryImage,  binaryImage,  binaryImage };
    cv::merge(channels, threeChannelImage);
    return threeChannelImage;
}

cv::Mat PictureMatch::img_BGR2GRAY3C(cv::Mat& img, double t1, double t2, int type) {
    cv::Mat grayImage;
    // ����ɫͼƬת��Ϊ�Ҷ�ͼƬ
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat binaryImage = cv::Mat::zeros(grayImage.size(), grayImage.type());
    cv::threshold(grayImage, binaryImage, t1, t2, type);
    cv::Mat binaryImage2 = cv::Mat::zeros(grayImage.size(), grayImage.type());
    cv::threshold(binaryImage, binaryImage2, 0, 255, cv::THRESH_BINARY);
    cv::Mat threeChannelImage;
    std::vector<cv::Mat> channels = { binaryImage2,  binaryImage2,  binaryImage2 };
    cv::merge(channels, threeChannelImage);
    return threeChannelImage;
}

double PictureMatch::calculate_brightness(cv::Mat& image) {
    // ��ͼƬ��BGRת����HSVɫ�ʿռ�
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // ����Vֵ��ƽ��ֵ��ΪͼƬ�������̶�
    double meanV = cv::mean(hsvImage)[2];

    return meanV;
}

double PictureMatch::calculate_blue_ratio(const cv::Mat& image) {
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    // �趨��ɫ����ֵ��Χ
    // ע�⣺HSV�е�ɫ��(H)�����Ͷ�(S)������(V)�ķ�Χ������ͼƬ����
    int low_h = 80;  // ��ɫ���HSV��Hֵ
    int high_h = 145; // ��ɫ���HSV��Hֵ
    int low_s = 0;   // ��ͱ��Ͷ�
    int high_s = 255; // ��߱��Ͷ�
    int low_v = 0;    // �������
    int high_v = 255; // �������

    // ������ֵ������Ĥ
    cv::Mat mask;
    cv::inRange(hsv_image, cv::Scalar(low_h, low_s, low_v), cv::Scalar(high_h, high_s, high_v), mask);
    //cv::imwrite("1234858.jpg", mask);
    // ������ɫ�������������������������
    double blue_pixels = cv::countNonZero(mask);
    double total_pixels = image.size().area();
    //std::cout << mask.channels() << std::endl;
    //std::cout << blue_pixels << std::endl;
    //std::cout << total_pixels << std::endl;
    // ������ɫ�ص�ռ��
    return blue_pixels / total_pixels;
}