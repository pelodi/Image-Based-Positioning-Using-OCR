package com.example.opencv;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Camera;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.SparseArray;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;


import com.example.opencv.GraphicOverlay;
import com.example.opencv.OcrGraphic;
import com.google.android.gms.common.util.CollectionUtils;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.Text;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class EdgeDetection extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {


    @Nullable
    private AsyncTask<Void, Void, Boolean> textDetectAsyncTask;

    String directory = System.getProperty("user.home");
    String fileName = "results.txt";
    String absolutePath = directory + File.separator + fileName;

    private TextRecognizer textRecognizer;

    private GraphicOverlay<OcrGraphic> graphicOverlay;

    private TextView textViewValue;
    private TextView textViewValue2;
    private static Scalar CONTOUR_COLOR = null;
    private static double areaThreshold = 1.025; //threshold for the area size of an object
    private static final String TAG = "EdgeDetection";
    private CameraBridgeViewBase cameraBridgeViewBase;
    private CameraBridgeViewBase.CvCameraViewListener2 cameraViewListener;

    private SparseArray<TextBlock> items;


    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    cameraBridgeViewBase.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };


    public static Intent newIntent(@NonNull Context context) {
        return new Intent(context, EdgeDetection.class);
    }

    public static void start(Context context) {
        Intent starter = new Intent(context, EdgeDetection.class);
        context.startActivity(starter);
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_open_cvcamera);

        Log.d("EdgeDetectionClass", "onCreate");

        cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        graphicOverlay = (GraphicOverlay<OcrGraphic>) findViewById(R.id.graphicOverlay);
        textRecognizer = new TextRecognizer.Builder(this).build();
        textViewValue = findViewById(R.id.text_value);

    }

    @Override
    public void onPause() {
        super.onPause();

        textDetectAsyncTask.cancel(true);

        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();

        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, baseLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        textDetectAsyncTask.cancel(true);


        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }

    }


    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        final Mat color = inputFrame.rgba();
        final Mat edges = inputFrame.gray();
        //Imgproc.Canny(edges,edges,50,200,3);
        Mat hierarchy = new Mat();
        final List<MatOfPoint> contours = new ArrayList<MatOfPoint>();


        Mat edgesp = edges.clone();
        Imgproc.cvtColor(edges, edgesp, Imgproc.COLOR_GRAY2BGR);


        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        double maxArea = -1;
        int maxAreaIdx = -1;


        //Null Check for contours
        if (contours != null) {
            if (!contours.isEmpty()) {
                MatOfPoint temp_contour = contours.get(0); //the largest is at the index 0 for starting point
                MatOfPoint2f approxCurve = new MatOfPoint2f();
                MatOfPoint largest_contour = contours.get(0);

                List<MatOfPoint> largest_contours = new ArrayList<MatOfPoint>();

                for (int idx = 0; idx < contours.size(); idx++) {
                    temp_contour = contours.get(idx);
                    double contourarea = Imgproc.contourArea(temp_contour);
                    //compare this contour to the previous largest contour found
                    if (contourarea > maxArea) {
                        //check if this contour is a square
                        MatOfPoint2f new_mat = new MatOfPoint2f(temp_contour.toArray());
                        int contourSize = (int) temp_contour.total();
                        MatOfPoint2f approxCurve_temp = new MatOfPoint2f();
                        Imgproc.approxPolyDP(new_mat, approxCurve_temp, contourSize * 0.05, true);
                        if (approxCurve_temp.total() == 4) {
                            maxArea = contourarea;
                            maxAreaIdx = idx;
                            approxCurve = approxCurve_temp;
                            largest_contour = temp_contour;
                        }
                    }
                }

                //Convert back to MatOfPoint
                final MatOfPoint points = new MatOfPoint(approxCurve.toArray());
                final Rect rect = Imgproc.boundingRect(points);


                textDetectAsyncTask = new TextDetectAsyncTask(contours, color, edges, rect);
                textDetectAsyncTask.execute();


                Mat src = lineDetection(color);
                //Imgproc.rectangle(color, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(250, 0, 0, 255), 3);
                return src;

            } else {
                return color;
            }
        } else {
            return color;
        }

    }

    private Mat lineDetection(Mat img) {
        Mat hsv = new Mat();
        Mat color_range_red = new Mat();
        Mat mask = new Mat();
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV );
        Core.inRange(hsv, new Scalar(30,150,50), new Scalar(255,255,180), color_range_red);
        Core.bitwise_and(img, img, mask, color_range_red);
        //***********convert to grayscale
        Mat gray_mask = new Mat();
        Imgproc.cvtColor(mask, gray_mask, Imgproc.COLOR_BGR2GRAY);
        return mask;
    }

    private double maxValue(double[] lengthArray) {
        double max = lengthArray[0];
        for(int i = 1; i < lengthArray.length; i++) {
            if(lengthArray[i] > max) {
                max = lengthArray[i];
            }
        }
        return max;
    }

    private double minValue(double[] pointArray) {
        double min = pointArray[0];
        for(int i = 1; i < pointArray.length; i++) {
            if (pointArray[i] < min) {
                min = pointArray[i];
            }
        }
        return min;
    }


    @SuppressLint("StaticFieldLeak")
    private class TextDetectAsyncTask extends AsyncTask<Void, Void, Boolean> {
        private final List<MatOfPoint> contours;
        private final Rect rect;
        private final Mat color;
        private final Mat edges;

        TextDetectAsyncTask(final List<MatOfPoint> contours, final Mat color, final Mat edges, final Rect rect) {
            this.contours = contours;
            this.rect = rect;
            this.color = color;
            this.edges = edges;
        }


        private void prepareData() {
            //VANISHING POINT DETECTION
            Mat lines = new Mat();
            //Mat blurImg = new Mat();
            Mat src = lineDetection(color);
            Mat gray_mask = new Mat();
            Imgproc.cvtColor(src, gray_mask, Imgproc.COLOR_BGR2GRAY);
            Imgproc.HoughLinesP(gray_mask, lines,1, Math.PI / 180, 100 , 520, 0);
            Log.d("lines", "count " + lines.rows());

            // Draw the lines
            /*for (int x = 0; x < lines.rows(); x++) {
                double[] l = lines.get(x, 0);
                Imgproc.line(mask, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
            }*/


            double[] ArrayA = new double[lines.rows()];
            double[] ArrayB = new double[lines.rows()];
            double[] ArrayC = new double[lines.rows()];
            double[] ArrayE = new double[lines.rows()];
            double[] ArrayF = new double[lines.rows()];
            double[] gamaArray = new double[lines.rows()];
            double[] lengthArray = new double[lines.rows()];
            double sumA = 0;
            double sumB = 0;
            double sumC = 0;
            double sumD = 0;
            double sumE = 0;
            double sumF = 0;

            for (int row = 0; row < lines.rows(); row++) {
                double[] l = lines.get(row, 0);
                Point p1 = new Point(l[0], l[1]);
                Point p2 = new Point(l[2], l[3]);

                double slope = (p2.y - p1.y) / (double) (p2.x - p1.x);
                double length = (double) Math.sqrt(Math.pow((p2.x - p1.x), 2) + Math.pow((p2.y - p1.y), 2));
                double gama = Math.pow(slope, 2) + 1;
                double constant = ((slope * p1.x) - p1.y) * (-1);
                double A = (1 / gama) * length * Math.pow(slope, 2);
                double B = (1 / gama) * length * slope * (-1);
                double C = (1 / gama) * length * slope * constant;
                double D = constant;
                double E = (1 / gama) * length;
                double F = (1 / gama) * length * (-1) * constant;
                /*Log.d("F", "F found! " + F);
                Log.d("E", "E found! " + E);
                Log.d("D", "D found! " + D);
                Log.d("C", "C found! " + C);
                Log.d("B", "B found! " + B);
                Log.d("A", "A found! " + A);
                Log.d("gama", "Gama found! " + gama);
                Log.d("constant", "Constant found! " + constant);
                Log.d("slope", "Slope found! " + slope);*/
                double[] slopeArray = new double[lines.rows()];
                slopeArray[row] = slope;
                gamaArray[row] = gama;
                ArrayA[row] = A;
                ArrayB[row] = B;
                ArrayC[row] = C;
                ArrayE[row] = E;
                ArrayF[row] = F;
                lengthArray[row] = length;
            }


            //Sum of the ArrayA
            for (int i = 0; i < ArrayA.length; i++) {
                sumA = sumA + ArrayA[i];
            }
            //Log.d("Summation", "sumA found! " + sumA);

            //Sum of the ArrayB
            for (int i = 0; i < ArrayB.length; i++) {
                sumB = sumB + ArrayB[i];
            }
            //Log.d("Summation", "sumB found! " + sumB);

            //Sum of the ArrayC
            for (int i = 0; i < ArrayC.length; i++) {
                sumC = sumC + ArrayC[i];
            }
            //Log.d("Summation", "sumC found! " + sumC);

            //Sum of the ArrayE
            for (int i = 0; i < ArrayE.length; i++) {
                sumE = sumE + ArrayE[i];
            }
            //Log.d("Summation", "sumE found! " + sumE);

            //Sum of ArrayF
            for (int i = 0; i < ArrayF.length; i++) {
                sumF = sumF + ArrayF[i];
            }
            //Log.d("Summation", "sumF found! " + sumF);

            //sumD = sumB;

            float VPx = (float) ((float)  (((sumB * sumF) - (sumC * sumE)) / ((sumA * sumE) - (sumB * sumB))) );
            float VPxprime = VPx - (1000 / 2);
            Log.d("Vanishing Point x", "VPx found! ********************************************" + VPx);


            float theta = (float) ((-71000) / VPxprime);
            Log.d("Theta", "Theta found! *******************************************" + theta);

            double[] p1xArray = new double[lines.rows()];
            double[] p1yArray = new double[lines.rows()];
            double[] p2xArray = new double[lines.rows()];
            double[] p2yArray = new double[lines.rows()];
            double[] slopeArray2 = new double[lines.rows()];
            for(int y = 0; y < lines.rows(); y++) {
                double[] l = lines.get(y, 0);
                Point p1 = new Point(l[0], l[1]);
                Point p2 = new Point(l[2], l[3]);
                double slope = (p2.y - p1.y) / (double) (p2.x - p1.x);
                slopeArray2[y] = slope;
                p1xArray[y] = p1.x;
                p1yArray[y] = p1.y;
                p2xArray[y] = p2.x;
                p2yArray[y] = p2.y;
                Log.d("endpoints", "p1,p2" + p1 + p2);
            }

            double x1Max = 0;
            double x1Min = 0;
            double y1Max = 0;
            double y1Min = 0;
            double x2Max = 0;
            double x2Min = 0;
            double y2Max = 0;
            double y2Min = 0;
            double length1 = 0;
            double length2 = 0;
            Point k = new Point(0,0);
            Point m = new Point(0,0);
            Point l = new Point(0,0);
            Point n = new Point(0,0);
            for(int x = 0; x < slopeArray2.length; x++) {
                x1Max = maxValue(p1xArray);
                y1Max= maxValue(p1yArray);
                x1Min = minValue(p1xArray);
                y1Min = minValue(p1yArray);
                x2Max = maxValue(p2xArray);
                y2Max = maxValue(p2yArray);
                x2Min = minValue(p2xArray);
                y2Min = minValue(p2yArray);
                if(slopeArray2[x] < 0) {
                    k = new Point(x1Min, y1Max);
                    m = new Point(x2Max, y2Min);
                    length1 = Math.sqrt(Math.pow((m.x - k.x), 2) + Math.pow((m.y - k.y), 2));
                }
                else {
                    l = new Point(x1Min, y1Min);
                    n = new Point(x2Max, y2Max);
                    length2 = Math.sqrt(Math.pow((n.x - l.x), 2) + Math.pow((n.y - l.y), 2));
                }
            }

            Log.d("p1x array", "P1x" + p1xArray);
            Log.d("p1y array", "P1y" + p1yArray);
            Log.d("p2x array", "P2x" + p2xArray);
            Log.d("p2y array", "P2y" + p2yArray);


            Log.d("points", "points" + k + m + l + n);

            Log.d("points", "points" + length1 + length2);

            double maxLength = 0;
            double avgLength = 0;
            if(length1 > length2) {
                maxLength = length1;
            }
            else {
                maxLength = length2;
            }

            float depth = (float) ((0.5 * Math.cos(theta) * (20 / maxLength) * (1 + Math.sqrt(1 + (maxLength * maxLength * Math.pow(Math.tan(theta), 2))))));
            Log.d("depth", "distance" + depth);


            Imgproc.cvtColor(edges, edges, Imgproc.COLOR_BayerBG2RGB);

            new Thread(new Runnable() {
                @Override
                public void run() {

                    for (int j = 0; j < contours.size(); j++) {
                        Point bottomRight = rect.br();
                        Point topLeft = rect.tl();


                        Mat subImage = color.submat((int) topLeft.y, (int) bottomRight.y, (int) topLeft.x, (int) bottomRight.x);
                        if (subImage.width() > 0 && subImage.height() > 0) {

                            Bitmap bitMap = Bitmap.createBitmap(subImage.width(), subImage.height(), Bitmap.Config.ARGB_8888);
                            Utils.matToBitmap(subImage, bitMap);
                            Frame frame = new Frame.Builder().setBitmap(bitMap).build();
                            graphicOverlay.clear();
                            items = textRecognizer.detect(frame);
                            if (items != null) {
                                int size = items.size();
                                for (int i = 0; i < size; ++i) {
                                    final TextBlock item = items.valueAt(i);
                                    if (item != null) {

                                        Log.d("OcrDetectorProcessor", "Text detected! **************************************" + item.getValue());
                                        OcrGraphic graphic = new OcrGraphic(graphicOverlay, item);
                                        graphicOverlay.add(graphic);


                                        runOnUiThread(new Runnable() {
                                            @Override
                                            public void run() {
                                                textViewValue.setText(item.getValue());
                                            }
                                        });

                                    }
                                }
                            }

                        }
                    }


                }
            }).start();


        }

            @SuppressLint("WrongThread")
        @Override
        protected Boolean doInBackground(final Void... voids) {

            prepareData();
            return true;
        }

        @Override
        protected void onPostExecute(final Boolean success) {
            // if you need some other type you'd like to use in UI change type
        }
    }

}

