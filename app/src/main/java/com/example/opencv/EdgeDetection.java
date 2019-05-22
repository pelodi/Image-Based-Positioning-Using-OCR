package com.example.opencv;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
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
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


public class EdgeDetection extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {


    @Nullable
    private AsyncTask<Void, Void, Boolean> textDetectAsyncTask;


    //private final OcrDetectorListener ocrDetectorListener;
    private TextRecognizer textRecognizer;

    private GraphicOverlay<OcrGraphic> graphicOverlay;

    private TextView textViewValue;
    private static Scalar CONTOUR_COLOR = null;
    private static double areaThreshold = 1.025; //threshold for the area size of an object
    private static final String TAG = "EdgeDetection";
    private CameraBridgeViewBase cameraBridgeViewBase;
    private CameraBridgeViewBase.CvCameraViewListener2 cameraViewListener;

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

    public interface OcrDetectorListener {
        //void onTextDetected(String chassis);
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        final Mat color = inputFrame.rgba();
        final Mat edges = inputFrame.gray();
        Mat hierarchy = new Mat();
        final List<MatOfPoint> contours = new ArrayList<MatOfPoint>();


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

        /*Mat lines = new Mat();
        Mat edgesp = edges.clone();
        Imgproc.cvtColor(edges, edgesp, Imgproc.COLOR_GRAY2BGR);
        Imgproc.HoughLinesP(edges, lines, 1, Math.PI/180, 100, 50, 10); // runs the actual detection
        // Draw the lines
        for (int x = 0; x < lines.rows(); x++) {
            double[] l = lines.get(x, 0);
            Imgproc.line(edgesp, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(0, 0, 255), 3, Imgproc.LINE_AA, 0);
        }*/

                Imgproc.rectangle(color, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(250, 0, 0, 255), 3);
                return color;

            } else {
                return color;
            }
        } else {
            return color;
        }

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

        @Override
        protected Boolean doInBackground(final Void... voids) {

            Imgproc.cvtColor(edges, edges, Imgproc.COLOR_BayerBG2RGB);

            //GMS VISION DENEMESİ

            for (int j = 0; j < contours.size(); j++) {
                Point bottomRight = rect.br();
                Point topLeft = rect.tl();


                Mat subImage = color.submat((int) topLeft.y, (int) bottomRight.y, (int) topLeft.x, (int) bottomRight.x);
                if (subImage.width() > 0 && subImage.height() > 0) {

                    Bitmap bitMap = Bitmap.createBitmap(subImage.width(), subImage.height(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(subImage, bitMap);
                    Frame frame = new Frame.Builder().setBitmap(bitMap).build();
                    graphicOverlay.clear();
                    SparseArray<TextBlock> items = textRecognizer.detect(frame);
                    if (items != null) {
                        int size = items.size();
                        for (int i = 0; i < size; ++i) {
                            final TextBlock item = items.valueAt(i);
                            if (item != null) {

                                Log.d("OcrDetectorProcessor", "Text detected! " + item.getValue());
                                OcrGraphic graphic = new OcrGraphic(graphicOverlay, item);
                                graphicOverlay.add(graphic);


                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {


                                        new Handler().postDelayed(new Runnable() {
                                            @Override
                                            public void run() {
                                                textViewValue.setText(item.getValue());
                                            }
                                        }, 2000);

                                    }
                                });
                            }
                        }
                    }
                }

            }

            return true;
        }

        @Override
        protected void onPostExecute(final Boolean success) {
            // if you need some other type you'd like to use in UI change type
        }
    }

}

