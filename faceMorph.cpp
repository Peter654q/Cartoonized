#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <unistd.h>

using namespace cv;
using namespace std;
void joyful(vector<Point2f> &p2);
void angry(vector<Point2f> &p2);
void sad(vector<Point2f> &p2);
void confuse(vector<Point2f> &p2);

// Read points stored in the text files
vector<Point2f> readPoints(string pointsFileName, float width, float height)
{
    vector<Point2f> points;
    fstream fin;
    
    fin.open(pointsFileName.c_str(), ios::in);
    float trash;
    for(int i=0;i<3;i++){
        fin >> trash;
    }
    float facepoint[68][2];//facial point
    for(int i=0;i<2;i++){
        for(int j=0;j<68;j++){
            fin >> facepoint[j][i];
        }
    }
    for(int j=0;j<68;j++){
        float x=facepoint[j][0];
        float y=facepoint[j][1];
        points.push_back(Point2f(x,y));
    }
    //push corner point
    points.push_back(Point2f(0,0));
    points.push_back(Point2f(width-1,0));
    points.push_back(Point2f(0,height-1));
    points.push_back(Point2f(width-1,height-1));
    float x = points[27].x;
    float y = points[27].y-(points[33].y-points[27].y);
    points.push_back(Point2f(x, y));

    return points;
}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
    
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( srcTri, dstTri );
    
    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void morphTriangle(Mat &img1, Mat &img2, Mat &img, vector<Point2f> &t1, vector<Point2f> &t2, vector<Point2f> &t, double alpha)
{
    
    // Find bounding rectangle for each triangle
    Rect r = boundingRect(t);
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);
    
    // Offset points by left top corner of the respective rectangles
    vector<Point2f> t1Rect, t2Rect, tRect;
    vector<Point> tRectInt;
    for(int i = 0; i < 3; i++)
    {
        tRect.push_back( Point2f( t[i].x - r.x, t[i].y -  r.y) );
        tRectInt.push_back( Point(t[i].x - r.x, t[i].y - r.y) ); // for fillConvexPoly
        
        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
    }
    
    // Get mask by filling triangle
    Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
    fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
    
    // Apply warpImage to small rectangular patches
    Mat img1Rect, img2Rect;
    img1(r1).copyTo(img1Rect);
    img2(r2).copyTo(img2Rect);
    
    Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
    Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());
    
    applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);
    applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);
    
    // Alpha blend rectangular patches
    Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;
    
    // Copy triangular region of the rectangular patch to the output image
    multiply(imgRect,mask, imgRect);
    multiply(img(r), Scalar(1.0,1.0,1.0) - mask, img(r));
    img(r) = img(r) + imgRect;
    
    
}

Mat OilPaint(Mat I,int brushSize, int coarseness)
{
    assert(!I.empty());
    if (brushSize < 1) brushSize = 1;
    if (brushSize > 8) brushSize = 8;

    if (coarseness < 1) coarseness = 1;
    if (coarseness > 255) coarseness = 255;

    int width  = I.cols;
    int height = I.rows;

    int lenArray = coarseness + 1;
    int* CountIntensity = new int[lenArray];
    uint* RedAverage    = new uint[lenArray];
    uint* GreenAverage  = new uint[lenArray];
    uint* BlueAverage   = new uint[lenArray];

    //convert image to gray level
    Mat gray;
    cvtColor(I,gray,COLOR_BGR2GRAY);

    Mat dst = Mat::zeros(I.size(),I.type());

    for(int nY = 0;nY <height; nY++)
    {
        // set range of top and bottom
        int top = nY - brushSize;
        int bottom = nY+ brushSize+1;

        if(top<0) top = 0;
        if(bottom >=height) bottom = height - 1;

        for(int nX = 0;nX<width;nX++)
        {
            // set range of left and right
            int left = nX - brushSize;
            int right = nX +brushSize+1;

            if(left<0) left = 0;
            if(right>=width) right = width - 1;

            //init
            for(int i = 0;i <lenArray;i++)
            {
                CountIntensity[i] = 0;
                RedAverage[i] = 0;
                GreenAverage[i] = 0;
                BlueAverage[i] = 0;
            }

            // Rendering
            for(int j = top;j<bottom;j++)
            {
                for(int i = left;i<right;i++)
                {
                    uchar intensity = static_cast<uchar>(coarseness*gray.at<uchar>(j,i)/255.0);
                    CountIntensity[intensity]++;

                    RedAverage[intensity]  += I.at<Vec3b>(j,i)[2];
                    GreenAverage[intensity]+= I.at<Vec3b>(j,i)[1];
                    BlueAverage[intensity] += I.at<Vec3b>(j,i)[0];
                }
            }

            // find maximum
            uchar chosenIntensity = 0;
            int maxInstance = CountIntensity[0];
            for(int i=1;i<lenArray;i++)
            {
                if(CountIntensity[i]>maxInstance)
                {
                    chosenIntensity = (uchar)i;
                    maxInstance = CountIntensity[i];
                }
            }

            dst.at<Vec3b>(nY,nX)[2] = static_cast<uchar>(RedAverage[chosenIntensity] / static_cast<float>(maxInstance));
            dst.at<Vec3b>(nY,nX)[1] = static_cast<uchar>(GreenAverage[chosenIntensity] /  static_cast<float>(maxInstance));
            dst.at<Vec3b>(nY,nX)[0] = static_cast<uchar>(BlueAverage[chosenIntensity] /  static_cast<float>(maxInstance));
        }

    }
    imshow("OilPaint",dst);
    imwrite("OilPaint.jpg",dst);
    waitKey();

    return dst;
}

int main( int argc, char* argv[])
{
    int facialexpression = 0;
    cout << "which facial expression do you want to change?" << endl;
    cout << "0: no expression" << endl;
    cout << "1: joyful" << endl;
    cout << "2: angry" << endl;
    cout << "3: sad" << endl;
    cout << "4: confuse" << endl;
    cin >> facialexpression;
    string filename1(argv[argc-1]);
    string filename2(argv[argc-1]);
    
    //alpha controls the degree of morph
    double alpha = 0.5;
    
    //Read input images
    Mat img1 = imread(filename1);
    Mat img2 = imread(filename2);
    
    //convert Mat to float data type
    img1.convertTo(img1, CV_32F);
    img2.convertTo(img2, CV_32F);
    
    //empty average image
    Mat imgMorph = Mat::zeros(img1.size(), CV_32FC3);
    
    
    //Read points
    vector<Point2f> points1 = readPoints( filename1 + ".txt", img1.cols, img1.rows);
    vector<Point2f> points2 = readPoints( filename2 + ".txt", img2.cols, img2.rows);
    vector<Point2f> points;

    /////////////////////////////////////////////////////////////////////////////////
    //eyebows
    for(int i=17;i<27;i++){
        points2[i].y = (points2[i].y)*0.95;
    }
    //eyes
    points2[37].y = (points2[37].y)*0.98;
    points2[38].y = (points2[38].y)*0.98;
    points2[41].y = (points2[41].y)*1.02;
    points2[40].y = (points2[40].y)*1.02;
    points2[43].y = (points2[43].y)*0.98;
    points2[44].y = (points2[44].y)*0.98;
    points2[47].y = (points2[47].y)*1.02;
    points2[46].y = (points2[46].y)*1.02;
    //nose
    //points2[27].y = (points2[27].y)*1.05;
    points2[28].y = (points2[28].y)*1.05;
    points2[31].x = (points2[31].x)*1.04;
    points2[32].x = (points2[32].x)*1.04;
    points2[34].x = (points2[34].x)*0.96;
    points2[35].x = (points2[35].x)*0.96;
    //face shape
    points2[3].x = (points2[3].x)*1.03;
    points2[3].y = (points2[3].y)*0.97;
    points2[4].x = (points2[4].x)*1.03;
    points2[4].y = (points2[4].y)*0.97;
    points2[5].x = (points2[5].x)*1.03;
    points2[5].y = (points2[5].y)*0.97;
    points2[6].x = (points2[6].x)*1.03;
    points2[6].y = (points2[6].y)*0.97;
    points2[7].x = (points2[7].x)*1.03;
    points2[7].y = (points2[7].y)*0.97;
    points2[8].y = (points2[8].y)*0.97;
    points2[9].x = (points2[9].x)*0.95;
    points2[9].y = (points2[9].y)*0.98;
    points2[10].x = (points2[10].x)*0.97;
    points2[10].y = (points2[10].y)*0.97;
    points2[11].x = (points2[11].x)*0.97;
    points2[11].y = (points2[11].y)*0.97;
    points2[12].x = (points2[12].x)*0.98;
    points2[12].y = (points2[12].y)*0.97;
    points2[13].x = (points2[13].x)*0.99;
    points2[13].y = (points2[13].y)*0.97;
    //mouth
    for(int i=48;i<68;i++){
        points2[i].y = (points2[i].y)*0.98;
    }
    points2[48].x = (points2[48].x)*1.02;
    points2[54].x = (points2[54].x)*0.98;

    if(facialexpression==1){
        joyful(points2);
    }else if(facialexpression==2){
        angry(points2);
    }else if(facialexpression==3){
        sad(points2);
    }else if(facialexpression==4){
        confuse(points2);
    }
    /////////////////////////////////////////////////////////////////////////////////
    
    //compute weighted average point coordinates
    for(int i = 0; i < points1.size(); i++)
    {
        float x, y;
        x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
        y =  ( 1 - alpha ) * points1[i].y + alpha * points2[i].y;
        
        //points.push_back(Point2f(x,y));
        points.push_back(Point2f(points2[i].x,points2[i].y));
    }
    
    
    //Read triangle indices
    ifstream ifs("tri.txt");
    int x,y,z;
    Mat polydraw;
    img1.convertTo(polydraw, CV_8UC3);

    while(ifs >> x >> y >> z)
    {
        // Triangles
        if(x<73 && y<73 && z<73){
            vector<Point2f> t1, t2, t;
        
            // Triangle corners for image 1.
            t1.push_back( points1[x] );
            t1.push_back( points1[y] );
            t1.push_back( points1[z] );
            
            // Triangle corners for image 2.
            t2.push_back( points2[x] );
            t2.push_back( points2[y] );
            t2.push_back( points2[z] );
            
            // Triangle corners for morphed image.
            t.push_back( points[x] );
            t.push_back( points[y] );
            t.push_back( points[z] );
            
            morphTriangle(img1, img2, imgMorph, t1, t2, t, 0);

            Point polypoint[1][3];
            polypoint[0][0] = points1[x];
            polypoint[0][1] = points1[y];
            polypoint[0][2] = points1[z];
            const Point* ppt[1] = {polypoint[0]};
            int npt[] = {3};
            polylines(polydraw, ppt, npt, 1, 1, Scalar(0,255,255),3);
        }
        
        
    }
    
    Mat transformed, gray, edges_canny,edges_lap,edges_sobel, mask, hsvframe, small, small_temp, temp, cartoonized;
    imgMorph.convertTo(transformed, CV_8UC3);
    cvtColor(transformed, gray, CV_BGR2GRAY);
    //sobel , eyebows part disappear
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges_sobel);
    //threshold(edges_sobel, mask, 80, 255, THRESH_BINARY_INV);
    imshow( "edges_sobel", edges_sobel );
    imwrite( "edges_sobel.jpg", edges_sobel );
    //sobel

    //canny , too many little edges 
    GaussianBlur(transformed, transformed, Size(3,3), 0, 0);
    Canny(transformed, edges_canny, 50, 150, 3);
    //threshold(edges_canny, mask, 80, 255, THRESH_BINARY_INV);
    imshow("edges_canny",edges_canny);
    imwrite("edges_canny.jpg", edges_canny );
    //canny

    //laplacian , seems the best 
    medianBlur(gray, gray, 11);
    Laplacian(gray, edges_lap, CV_8U, 5);
    threshold(edges_lap, mask, 80, 255, THRESH_BINARY_INV);
    imshow("edges_lap",edges_lap);
    imwrite("edges_lap.jpg",edges_lap);
    //laplacian
    

    resize(transformed, small, cv::Size((transformed.cols)/4, (transformed.rows)/4), 0, 0, INTER_LINEAR);
    
    int repetitions = 7;
    for (int i = 0; i < repetitions; i++) {
        int kernelsize = 7;
        double sigmacolor = 18;
        double sigmaspace = 10;
        bilateralFilter(small, small_temp, kernelsize, sigmacolor, sigmaspace);
        bilateralFilter(small_temp, small, kernelsize, sigmacolor, sigmaspace);
    }

    resize(small, temp, cv::Size(transformed.cols, transformed.rows), 0, 0, INTER_LINEAR);
    cartoonized.setTo(0);
    temp.copyTo(cartoonized, mask);

    imshow("facemesh", polydraw);
    imshow("Morphed Face", cartoonized);
    imwrite("Mesh.jpg", polydraw);
    imwrite("Result0.jpg", transformed);
    imwrite("Result1.jpg", cartoonized);
    OilPaint(transformed,5,255);
    waitKey(0);
    
    return 0;
}

void joyful(vector<Point2f> &p2){
    if(p2[48].y>p2[49].y){
        p2[48].y=p2[48].y*0.97;
    }
    if(p2[54].y>p2[53].y){
        p2[54].y=p2[54].y*0.97;
    }
    for(int i=18;i<21;i++){
        p2[i].y = p2[i].y*0.95;
    }
    for(int i=23;i<26;i++){
        p2[i].y = p2[i].y*0.95;
    }
}

void sad(vector<Point2f> &p2){
    p2[48].y=p2[48].y*1.02;
    p2[54].y=p2[54].y*1.02;
    p2[21].y=p2[21].y*0.98;
    p2[22].y=p2[22].y*0.98;
    p2[17].y=p2[17].y*1.15;
    p2[18].y=p2[18].y*1.10;
    p2[26].y=p2[26].y*1.15;
    p2[25].y=p2[25].y*1.10;
    p2[36].y=p2[36].y*1.02;
    p2[45].y=p2[45].y*1.02;
    p2[67].y=p2[61].y;
    p2[66].y=p2[62].y;
    p2[65].y=p2[63].y;
}

void angry(vector<Point2f> &p2){
    p2[21].x=p2[21].x*1.05;
    p2[21].y=p2[21].y*1.1;
    p2[22].x=p2[22].x*0.95;
    p2[22].y=p2[22].y*1.1;
    p2[17].y=p2[17].y*0.95;
    p2[26].y=p2[26].y*0.95;
    p2[48].y=p2[48].y*1.02;
    p2[48].x=p2[48].x*0.98;
    p2[54].y=p2[54].y*1.02;
    p2[54].x=p2[54].x*1.02;
    p2[67].y=p2[61].y;
    p2[66].y=p2[62].y;
    p2[65].y=p2[63].y;
}

void confuse(vector<Point2f> &p2){  
    p2[21].x=p2[21].x*1.05;
    p2[21].y=p2[21].y*1.1;
    p2[22].x=p2[22].x*0.95;
    p2[22].y=p2[22].y*0.95;
    p2[67].y=p2[61].y;
    p2[66].y=p2[62].y;
    p2[65].y=p2[63].y;
};
