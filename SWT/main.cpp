#include <iostream>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class Ray
{
public:
    Point start;
    Point end;
    float grad;
    float theta;
    vector<Point> points;

    Ray();
    Ray(Point p, float gradient, float t)
    {
        start = p;
        grad = gradient;
        theta = t;
        points.push_back(start);
    }

    vector<Point> get_next_points(int n)
    {
        vector<Point> pts;
        Point curr = points.back();
        Point2f next = curr;
        float dx =  cos(theta);
        float dy =  -sin(theta);
        int cnt = 0;
        while (cnt < n)
        {
            next.x = next.x + dx;
            next.y = next.y + dy;
            if ((int)next.x != curr.x || (int)next.y != curr.y)
            {
                cnt++;
                curr.x = (int)next.x;
                curr.y = (int)next.y;
                pts.push_back(next);
            }
        }
        return pts;
    }

    bool check_for_endpoint(float dq)
    {
        float temp_dq = dq, temp_theta = theta;
        if (dq < 0) temp_dq += 2*CV_PI;
        if (theta < 0) temp_theta += 2*CV_PI;
        if (temp_dq > CV_PI) temp_dq -= CV_PI;
        if (temp_theta > CV_PI) temp_theta -= CV_PI;

        if (abs(temp_dq - temp_theta) < CV_PI/10 || (abs(temp_dq + temp_theta) > CV_PI - CV_PI/10 && abs(temp_dq + temp_theta) < CV_PI + CV_PI/10))
            return true;
        else{
            //cout<<temp_dq*180/CV_PI<<"\t"<<temp_theta*180/CV_PI<<endl;
            return false;
        }
    }

    float get_length()
    {
        return sqrt((start.x-end.x)*(start.x-end.x) + (start.y-end.y)*(start.y-end.y));
    }
};

class SWT
{
    Mat ip_img;
    Mat gradX, gradY, edge_map, stroke_width, rays_img, refined_stroke_width;
    vector<Ray> rays;

public:

    SWT(Mat &image);
    void get_edge_map();
    void calculate_gradX();
    void calculate_gradY();
    void show_images();
    void get_rays();
    void get_swt();
    void median_filter_rays();
    int get_median(vector<Point>);
    void draw_ray(Ray , bool);
    void draw_rays();
};

SWT::SWT(Mat &image)
{
    ip_img = image;
    gradX = Mat::zeros(ip_img.size(), CV_32FC1);
    gradY = Mat::zeros(ip_img.size(), CV_32FC1);
    stroke_width = Mat(ip_img.size(), CV_8UC1, Scalar(255));
    refined_stroke_width = Mat(ip_img.size(), CV_8UC1, Scalar(255));
    rays_img = Mat::zeros(ip_img.size(), CV_8UC1);
}

void SWT::get_edge_map()
{
    GaussianBlur(ip_img, ip_img, Size(5, 5), 1.2, 1.2);
    Canny(ip_img, edge_map, 10, 50, 3);

}
void SWT::calculate_gradX()
{
    Mat kernel_grad_x = Mat::zeros(3,3,CV_32FC1);
    kernel_grad_x.at<float>(0,0) = -1.0/4;
    kernel_grad_x.at<float>(1,0) = -2.0/4;
    kernel_grad_x.at<float>(2,0) = -1.0/4;
    kernel_grad_x.at<float>(0,2) =  1.0/4;
    kernel_grad_x.at<float>(1,2) =  2.0/4;
    kernel_grad_x.at<float>(2,2) =  1.0/4;
    filter2D(ip_img, gradX, 5, kernel_grad_x, Point(1,1));
}

void SWT::calculate_gradY()
{
    Mat kernel_grad_y = Mat::zeros(3,3,CV_32FC1);
    kernel_grad_y.at<float>(0,0) =  1.0/4;
    kernel_grad_y.at<float>(0,1) =  2.0/4;
    kernel_grad_y.at<float>(0,2) =  1.0/4;
    kernel_grad_y.at<float>(2,0) = -1.0/4;
    kernel_grad_y.at<float>(2,1) = -2.0/4;
    kernel_grad_y.at<float>(2,2) = -1.0/4;
    filter2D(ip_img, gradY, 5, kernel_grad_y, Point(1,1));
}

void SWT::get_rays()
{
    for (int r = 0; r < ip_img.rows; r++)
        for (int c = 0; c < ip_img.cols; c++)
        {
            if (edge_map.at<uchar>(r, c) > 0)
            {
                //cout<<"Found Edge Pixel ("<<r<<", "<<c<<")"<<endl;
                //cout<<i<<"\t"<<j<<"\t"<<(int)edge_map.at<uchar>(i, j)<<endl;
                float gx = gradX.at<float>(r, c);
                float gy = gradY.at<float>(r, c);

                float grad = sqrt(gx*gx + gy*gy);
                //if (grad == 0) continue;
                float theta = atan2(gy, gx);
                //cout<<gx<<"\t"<<gy<<"\t"<<theta<<endl;

                Ray ray = Ray(Point(c,r), grad, theta);
                while(true)
                {
                    bool end_found = false;
                    vector<Point> next = ray.get_next_points(210);
                    //cout<<"Potential next Point ("<<next.y<<", "<<next.x<<")"<<endl;
                    //cout<<"Started at :"<<ray.start.y<<"\t"<<ray.start.x<<endl;
                    //cout<<"Ray end at :"<<next.back().y<<"\t"<<next.back().x<<endl;
                    //cout<<"Ray direction: "<<ray.theta<<endl;
                    for (uint v = 0; v < next.size(); v++)
                    {
                        if (next[v].x > 0 && next[v].x <edge_map.cols && next[v].y > 0 && next[v].y <edge_map.rows)
                        {
                            ray.points.push_back(next[v]);
                            if (edge_map.at<uchar>(next[v].y, next[v].x) > 0)
                            {
                                //cout<<"Found Edge Pixel Again("<<next[v].y<<", "<<next[v].x<<")"<<endl;
                                end_found = true;
                                float gradinX = gradX.at<float>(next[v].y, next[v].x);
                                float gradinY = gradY.at<float>(next[v].y, next[v].x);
                                //if (gradinX == 0 && gradinY == 0)
                                //    continue;
                                float dq = atan2(gradinY, gradinX);
                                //cout<<"Gradient at new edge location: "<<dq<<endl;
                                if (ray.check_for_endpoint(dq))
                                {
                                    ray.end = next[v];
                                    //cout<<"Found a ray !!! "<<endl;
                                    rays.push_back(ray);
                                    //draw_ray(ray, true);
                                }
                                else
                                //draw_ray(ray, false);
                                //imshow("rays", rays_img);
                                //waitKey(0);
                                break;
                            }
                        }
                        else
                        {// ray has gone beyond image size
                            end_found = true;
                            break;
                        }
                    }
                    if (end_found) break;
                }
            }
        }
}

void SWT::get_swt()
{
    for (uint r=0; r<rays.size(); r++)
    {
        vector<Point> points = rays[r].points;
        float length = rays[r].get_length();
        for (uint p=0; p<points.size(); p++)
        {
            stroke_width.at<uchar>(points[p].y, points[p].x) = std::min((int)stroke_width.at<uchar>(points[p].y, points[p].x), (int)length);
        }
    }
}

int SWT::get_median(vector<Point> pts)
{
    std::set<uchar> swt;
    for (uint p=0; p<pts.size(); p++)
        swt.insert(stroke_width.at<uchar>(pts[p].y, pts[p].x));

    int median_pos = swt.size();
    int count = 0;
    uchar median = 255;

    set<uchar>::iterator it = swt.begin();
    for (it = swt.begin(); it!=swt.end(); ++it){
        if (count == median_pos/2){
            median = *it;
            break;
        }
        count++;
    }
    return median;
}

void SWT::median_filter_rays()
{
    for (uint r=0; r<rays.size(); r++)
    {
        vector<Point> pts = rays[r].points;
        int median = get_median(pts);
        for (uint p = 0; p<pts.size(); p++)
            if (stroke_width.at<uchar>(pts[p].y, pts[p].x) > median)
                refined_stroke_width.at<uchar>(pts[p].y, pts[p].x) = median;
            else
                refined_stroke_width.at<uchar>(pts[p].y, pts[p].x) = stroke_width.at<uchar>(pts[p].y, pts[p].x);
    }
}

void SWT::draw_rays()
{
    for (uint r=0; r<rays.size(); r++)
    {
        vector<Point> points = rays[r].points;
        for (uint p=0; p<points.size(); p++)
            rays_img.at<uchar>(points[p].y, points[p].x) = 255;
    }
    imshow("rays", rays_img);
}

void SWT::draw_ray(Ray r, bool res)
{
    vector<Point> points = r.points;
    for (uint p=0; p<points.size(); p++)
        if (res)
            rays_img.at<uchar>(points[p].y, points[p].x) = 255;
        else
            rays_img.at<uchar>(points[p].y, points[p].x) = 128;
}

void SWT::show_images()
{
    imshow("image", ip_img);
    imshow("edge_map", edge_map);
    imshow("stroke_width", stroke_width);
    imshow("refined_stroke_width", refined_stroke_width);

    //imshow("edgeX", gradX);
    //imshow("edgeY", gradY);
    waitKey(0);
}

int main()
{
    String img_path = "images/O.jpg";
    Mat img = imread(img_path, 0);
    resize(img, img, Size(), 1, 1, INTER_CUBIC);
/*
    // testing using Black Box image
    img = Mat::zeros(400, 400, CV_8UC1);

    for (int r = 250; r<350; r++)
    for (int c = 150; c<250; c++){
        img.at<uchar>(r,c) = 255;
    }
*/
    if (true)
    // invert image
    for (int r=0; r<img.rows; r++)
    for (int c = 0; c<img.cols; c++)
        img.at<uchar>(r,c) = 255-img.at<uchar>(r,c);
    /*
    // Testing code for ray direction
    Mat black = Mat::ones(400, 400, CV_8UC1)*255;
    black.at<uchar>(200,200) = 0;
    float thetas[] = {0, CV_PI/2, CV_PI, -CV_PI/2};
    for (int j = 0; j < 4 ; j++){
        Ray r = Ray(Point(100,200), 10, thetas[j]);
        vector<Point> p = r.get_next_points(90);
        for (int i = 0; i<p.size(); i++)
            black.at<uchar>(p[i].y, p[i].x) = 0;

        imshow("bb", black);
        waitKey(0);
        cout<<j<<endl;
    }
    //imwrite("try.png", black);
*/
    cout <<"Initializing SWT ..."<<endl;
    SWT swt = SWT(img);
    swt.get_edge_map();
    cout<<"Calculating gradients ...\n";
    swt.calculate_gradX();
    swt.calculate_gradY();

    cout<<"calculating stroke widths ...\n";
    swt.get_rays();
    swt.get_swt();
    //swt.show_images();
    cout<<"refining stroke widths ...\n";
    swt.median_filter_rays();
    //swt.draw_rays();
    cout<<"Displaying Results ...\n";
    swt.show_images();
    cout<<"Finished !!!";


    return 0;
}
