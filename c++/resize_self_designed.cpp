#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <utility>

#define ROLL 20
using namespace std;
using namespace cv;

template<typename Func,typename... Args>
auto measure_time(Func&& func,Args&&... args)
{
    using ReturnType = decltype(func(std::forward<Args>(args)...));
    ReturnType result;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0;i < ROLL;++i)
    {
       result = std::forward<Func>(func)(std::forward<Args>(args)...);
    }
    auto end  = std::chrono::high_resolution_clock::now();
    double elpse_time = std::chrono::duration<double>(end - start).count()/ROLL;
    return std::make_pair(result,elpse_time);
}



Mat bilinearResize(const Mat& src,int new_width,int new_height)
{
    int src_width = src.cols;
    int src_height = src.rows;
    Mat dst(new_height,new_width,src.type());

    float x_ratio = static_cast<float>(src_width) / new_width;
    float y_ratio = static_cast<float>(src_height) / new_height;

    for(int y = 0 ;y < new_height;y++)
    {
        for(int x = 0;x < new_width;x++)
        {
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            int x1 = static_cast<int>(src_x); 
            int y1 = static_cast<int>(src_y);
            int x2 = min(x1 + 1,src_width - 1 );
            int y2 = min(y1 + 1,src_height - 1);

            float dx = src_x - x1;
            float dy = src_y - y1;

            Vec3b p1 = src.at<Vec3b>(y1,x1);
            Vec3b p2 = src.at<Vec3b>(y2,x1);
            Vec3b p3 = src.at<Vec3b>(y1,x2);
            Vec3b p4 = src.at<Vec3b>(y2,x2);

            Vec3b result;
            for(int c = 0;c < 3;++c)
            {
                float R1 = (1 - dx) * p1[c] + dx * p2[c];
                float R2 = (1 - dx) * p3[c] + dx * p4[c];
                result[c] = static_cast<uchar>((1 - dy) * R1 + dy * R2);
            }
            dst.at<Vec3b>(y,x) = result;
        }
    }
    return dst;
}

Mat bilinearResize_parallel_threads(const Mat& src,int new_width,int new_height)
{
    int src_width = src.cols;
    int src_height = src.rows;
    Mat dst(new_height,new_width,src.type());

    float x_ratio = static_cast<float>(src_width) / new_width;
    float y_ratio = static_cast<float>(src_height) / new_height;

    parallel_for_(Range(0,new_height),[&](const Range& range)
    {
        for(int y = range.start ;y < range.end;y++)
        {
            for(int x = 0;x < new_width;x++)
            {
                float src_x = x * x_ratio;
                float src_y = y * y_ratio;

                int x1 = static_cast<int>(src_x); 
                int y1 = static_cast<int>(src_y);
                int x2 = min(x1 + 1,src_width - 1 );
                int y2 = min(y1 + 1,src_height - 1);

                float dx = src_x - x1;
                float dy = src_y - y1;

                Vec3b p1 = src.at<Vec3b>(y1,x1);
                Vec3b p2 = src.at<Vec3b>(y2,x1);
                Vec3b p3 = src.at<Vec3b>(y1,x2);
                Vec3b p4 = src.at<Vec3b>(y2,x2);

                Vec3b result;
                for(int c = 0;c < 3;++c)
                {
                    float R1 = (1 - dx) * p1[c] + dx * p2[c];
                    float R2 = (1 - dx) * p3[c] + dx * p4[c];
                    result[c] = static_cast<uchar>((1 - dy) * R1 + dy * R2);
                }
                dst.at<Vec3b>(y,x) = result;
            }
        }
    });
    return dst;
                
}

int main()
{

    Mat src = imread("boqi.jpg");
    int x_radio = 2;
    int y_radio = 2;

    auto [resized_img,elpse_time] = measure_time(bilinearResize,src,src.cols * x_radio,src.rows * y_radio);

    std::cout<<"原始方法耗时："<<elpse_time * 1000<<"ms"<<std::endl;


    auto [resized_thread_img,elpse_time_v2] = measure_time(bilinearResize_parallel_threads,src,src.cols * x_radio,src.rows * y_radio);
    
    std::cout<<"多线程耗时"<<elpse_time_v2 * 1000<<"ms"<<std::endl;
    imshow("src",src);
    imshow("resized_img",resized_img);

    waitKey(0);

    imwrite("resized_img.jpg",resized_img);
    imwrite("resized_threads_img.jpg",resized_thread_img);

    return 0;
}