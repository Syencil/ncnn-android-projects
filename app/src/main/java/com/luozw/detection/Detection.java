package com.luozw.detection;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class Detection {
    static{
        System.loadLibrary("ncnn_detection");
    }

    public static native boolean init(AssetManager manager);
    public static native boolean convert(AssetManager manager);
    public static native Box[] detect(Bitmap bitmap, double threshold, double nms_threshold);
    public static native boolean setInferThreadNum(int threadNum);
    public static native boolean setProcessThreadNum(int threadNum);
    public static native int benchmark();
    public static native int convertSize();
}
