package com.luozw.detection;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private static int MODEL_TYPE = 0; // 0 yolo 1 face
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static final int REQUEST_PICK_IMAGE = 2;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    private ImageView resultImageView;
    private SeekBar nmsSeekBar;
    private SeekBar thresholdSeekBar;
    private TextView thresholdTextview;
    private double threshold = 0.45,nms_threshold = 0.5;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    this,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
        Detection.init(getAssets());
        resultImageView = findViewById(R.id.imageView);
        thresholdTextview = findViewById(R.id.valTxtView);
        nmsSeekBar = findViewById(R.id.nms_seek);
        thresholdSeekBar = findViewById(R.id.threshold_seek);
        thresholdTextview.setText(String.format(Locale.ENGLISH,"Thresh:%.2f,NMS:%.2f",threshold,nms_threshold));
        nmsSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                nms_threshold = i/100.f;
                thresholdTextview.setText(String.format(Locale.ENGLISH,"Thresh:%.2f,NMS:%.2f",threshold,nms_threshold));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
        thresholdSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                threshold = i/100.f;
                thresholdTextview.setText(String.format(Locale.ENGLISH,"Thresh:%.2f,NMS:%.2f",threshold,nms_threshold));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
        Button inference = findViewById(R.id.button);
        inference.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_PICK);
                intent.setType("image/*");
                startActivityForResult(intent, REQUEST_PICK_IMAGE);
            }
        });

        final Button convert = findViewById(R.id.button_convert);
        convert.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                boolean status = Detection.convert(getAssets());
                if (status){
                    if(MODEL_TYPE==0){
                        MODEL_TYPE = 1;
                    }else if(MODEL_TYPE==1){
                        MODEL_TYPE = 0;
                    }
                    Toast.makeText(MainActivity.this, "Convert Model...", Toast.LENGTH_SHORT).show();
                }
            }
        });

        final Button benchmark = findViewById(R.id.button_benchmark);
        benchmark.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                int fps = Detection.benchmark();
                Toast.makeText(MainActivity.this, "Run 10 case in FPS " + Integer.toString(fps), Toast.LENGTH_SHORT).show();
            }
        });

        final Button convertSize = findViewById(R.id.button_convertSize);
        convertSize.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                int size = Detection.convertSize();
                Toast.makeText(MainActivity.this, "Convert inputsize to " + Integer.toString(size), Toast.LENGTH_SHORT).show();
            }
        });

        final EditText editText1 = findViewById(R.id.inferThread);
        editText1.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                String txt = editText1.getText().toString();
                int num_threads;
                try{
                    num_threads = Integer.parseInt(txt);
                }catch (NumberFormatException e){
                    Toast.makeText(MainActivity.this, "Invalid NCNN Num Thread", Toast.LENGTH_SHORT).show();
                    num_threads = 1;
                }
                if(Detection.setInferThreadNum(num_threads)){
                    Toast.makeText(MainActivity.this, "Set NCNN Num Thread = " + txt, Toast.LENGTH_SHORT).show();
                }else{
                    Toast.makeText(MainActivity.this, "Set NCNN Num Thread Failed! Using default Params", Toast.LENGTH_SHORT).show();
                }
                return false;
            }
        });

        final EditText editText2 = findViewById(R.id.processThread);
        editText2.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                String txt = editText2.getText().toString();
                int num_threads;
                try{
                    num_threads = Integer.parseInt(txt);
                }catch (NumberFormatException e){
                    Toast.makeText(MainActivity.this, "Invalid Num Thread", Toast.LENGTH_SHORT).show();
                    num_threads = 1;
                }
                if(Detection.setProcessThreadNum(num_threads)){
                    Toast.makeText(MainActivity.this, "Set Process Num Thread = " + txt, Toast.LENGTH_SHORT).show();
                }else{
                    Toast.makeText(MainActivity.this, "Set Process Num Thread Failed! Using default Params", Toast.LENGTH_SHORT).show();
                }
                return false;
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for(int result:grantResults){
            if(result != PackageManager.PERMISSION_GRANTED){
                this.finish();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(data == null){
            return;
        }
        Bitmap image = getPicture(data.getData());

        long start = System.currentTimeMillis();
        Box[] result = Detection.detect(image,threshold,nms_threshold);
        long end = System.currentTimeMillis();
        long time = end - start;
        Toast.makeText(MainActivity.this, "time：" + time + "ms", Toast.LENGTH_SHORT).show();
        Bitmap mutableBitmap = image.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        final Paint boxPaint = new Paint();
        boxPaint.setAlpha(200);
        boxPaint.setStyle(Paint.Style.STROKE);
        float size = image.getWidth()/800;
        boxPaint.setStrokeWidth(4 * size);
        boxPaint.setTextSize(40 * size);
        for(Box box:result){
            boxPaint.setColor(box.getColor());
            boxPaint.setStyle(Paint.Style.FILL);
            if(MODEL_TYPE==0){
                canvas.drawText(box.getVOCLabel(),box.x0,box.y0,boxPaint);
            }else if(MODEL_TYPE==1){
                canvas.drawText(box.getFaceLabel(),box.x0,box.y0,boxPaint);
            }
            DecimalFormat decimalFormat=new DecimalFormat("0.00");
            String score = decimalFormat.format(box.getScore());

            canvas.drawText(score,box.x0,box.y0 + 40*size,boxPaint);
            boxPaint.setStyle(Paint.Style.STROKE);
            canvas.drawRect(box.getRect(),boxPaint);
        }
        resultImageView.setImageBitmap(mutableBitmap);
    }

    public Bitmap getPicture(Uri selectedImage) {
        String[] filePathColumn = { MediaStore.Images.Media.DATA };
        Cursor cursor = this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
        cursor.moveToFirst();
        int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
        String picturePath = cursor.getString(columnIndex);
        cursor.close();
        Bitmap bitmap = BitmapFactory.decodeFile(picturePath);
        int rotate = readPictureDegree(picturePath);
        return rotateBitmapByDegree(bitmap,rotate);
    }

    public int readPictureDegree(String path) {
        int degree = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degree = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degree = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degree = 270;
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return degree;
    }

    public Bitmap rotateBitmapByDegree(Bitmap bm, int degree) {
        Bitmap returnBm = null;
        Matrix matrix = new Matrix();
        matrix.postRotate(degree);
        try {
            returnBm = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(),
                    bm.getHeight(), matrix, true);
        } catch (OutOfMemoryError e) {
        }
        if (returnBm == null) {
            returnBm = bm;
        }
        if (bm != returnBm) {
            bm.recycle();
        }
        return returnBm;
    }


}