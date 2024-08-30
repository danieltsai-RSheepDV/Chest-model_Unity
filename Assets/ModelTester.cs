using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;

public class ModelTester : MonoBehaviour
{
    public NNModel modelAsset;
    public RawImage rawImage;

    private Model model;
    private IWorker worker;

    private WebCamTexture webCamTexture;
    private Texture2D texture2D;

    private List<float> sizes = new List<float>();
    private List<float> timestamps = new List<float>();
    private float startTime;

    private float lastChestSize = 0f;
    private bool isInhaling = false;
    private float breathingThreshold = 0.01f; // Adjust this value based on your needs
     // Assign this in the Unity Inspector // Assign this in the Unity Inspector

    // Define class labels
    private readonly string[] classLabels = { "chest", "head" };

    private bool isInitializing = false;

    // Define constants for the model input dimensions
    private const int MODEL_INPUT_WIDTH = 224;
    private const int MODEL_INPUT_HEIGHT = 224;
    private const int MODEL_INPUT_CHANNELS = 3;

    private void PrintModelInfo()
    {
        Debug.Log("Model info:");
        Debug.Log("Inputs:");
        foreach (var input in model.inputs)
        {
            Debug.Log($"  Name: {input.name}, Shape: [{string.Join(", ", input.shape)}]");
        }
        Debug.Log("Outputs:");
        foreach (var output in model.outputs)
        {
            Debug.Log($"  Name: {output}");
        }
    }

    void Start()
    {
        Debug.Log("Starting initialization...");

        if (modelAsset == null)
        {
            Debug.LogError("modelAsset is not assigned. Please assign it in the Inspector.");
            return;
        }

        try
        {
            model = ModelLoader.Load(modelAsset);
            if (model == null)
            {
                Debug.LogError("Failed to load model. ModelLoader.Load returned null.");
                return;
            }
            Debug.Log("Model loaded successfully.");
            PrintModelInfo();

            worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
            if (worker == null)
            {
                Debug.LogError("Failed to create worker. WorkerFactory.CreateWorker returned null.");
                return;
            }
            Debug.Log("Worker created successfully.");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Exception while initializing model or worker: {e.Message}");
            return;
        }

        WebCamDevice[] devices = WebCamTexture.devices;
        Debug.Log($"Number of webcam devices found: {devices.Length}");

        if (devices.Length > 0)
        {
            for (int i = 0; i < devices.Length; i++)
            {
                Debug.Log($"Webcam {i}: {devices[i].name}");
            }

            webCamTexture = new WebCamTexture(devices[0].name, 800, 608);

            if (webCamTexture != null)
            {
                rawImage.texture = webCamTexture;
                webCamTexture.Play();

                startTime = Time.time;

                Debug.Log("WebCamTexture initialized and started successfully.");
                isInitializing = true;
                StartCoroutine(InitializeTexture2D());
            }
            else
            {
                Debug.LogError("Failed to initialize WebCamTexture.");
            }
        }
        else
        {
            Debug.LogError("No webcam devices found.");
        }
    }

    private IEnumerator InitializeTexture2D()
    {
        Debug.Log("Starting InitializeTexture2D coroutine");
        Debug.Log($"Initial webcam texture dimensions: {webCamTexture.width}x{webCamTexture.height}");

        yield return new WaitUntil(() => webCamTexture.width > 16 && webCamTexture.height > 16);

        Debug.Log($"Webcam texture ready. Dimensions: {webCamTexture.width}x{webCamTexture.height}");

        texture2D = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGBA32, false);
        Debug.Log($"texture2D initialized successfully with dimensions: {texture2D.width}x{texture2D.height}");

        isInitializing = false;
    }

    void Update()
    {
        string s = WebCamTexture.devices[0].name;
    }

    private void LateUpdate()
    {
        if (webCamTexture == null || !webCamTexture.isPlaying)
        {
            Debug.LogWarning("webCamTexture is null or not playing");
            return;
        }

        if (isInitializing)
        {
            Debug.Log("Still initializing texture2D...");
            return;
        }

        if (texture2D == null)
        {
            Debug.LogWarning("texture2D is not yet initialized");
            return;
        }
        
        if (webCamTexture.didUpdateThisFrame)
        {
            print("test");
            
            // Update texture2D with new webcam frame
            Color32[] pixels = webCamTexture.GetPixels32();
            texture2D.SetPixels32(pixels);
            texture2D.Apply();

            // Resize the texture to match the model's input size
            Texture2D resizedTexture = ResizeTexture(texture2D, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

            // Create the input tensor
            using (var tensor = new Tensor(1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNELS))
            {
                // Fill the tensor with image data
                for (int yPos = 0; yPos < MODEL_INPUT_HEIGHT; yPos++)
                {
                    for (int xPos = 0; xPos < MODEL_INPUT_WIDTH; xPos++)
                    {
                        Color pixel = resizedTexture.GetPixel(xPos, yPos);
                        tensor[0, yPos, xPos, 0] = pixel.r;
                        tensor[0, yPos, xPos, 1] = pixel.g;
                        tensor[0, yPos, xPos, 2] = pixel.b;
                    }
                }

                // Execute the model
                worker.Execute(tensor);

                // Process the output
                var output = worker.PeekOutput();
                var (chestX, chestY, chestWidth, chestHeight, confidence) = ProcessModelOutput(output);

                if (confidence > 0.05f) // Adjust this threshold as needed
                {
                    float chestSize = CalculateChestSize(chestX, chestY, chestWidth, chestHeight);
                    DetectBreathing(chestSize);

                    // Visualize the bounding box
                    DrawBoundingBox(chestX, chestY, chestWidth, chestHeight);
                }
                else
                {
                    Debug.Log("No chest detected with high confidence");
                }
            }

            // Clean up
            Destroy(resizedTexture);
        }
    }

    private (float x, float y, float width, float height, float confidence) ProcessModelOutput(Tensor output)
    {
        float maxConfidence = 0f;
        float x = 0f, y = 0f, width = 0f, height = 0f;

        for (int i = 0; i < 1029; i++)
        {
            float confidence = output[0, 0, i, 4];
            if (confidence > maxConfidence)
            {
                maxConfidence = confidence;
                x = output[0, 0, i, 0];
                y = output[0, 0, i, 1];
                width = output[0, 0, i, 2];
                height = output[0, 0, i, 3];
            }
        }

        return (x, y, width, height, maxConfidence);
    }

    private float CalculateChestSize(float x, float y, float width, float height)
    {
        // Calculate the area of the bounding box
        float area = width * height;

        // Calculate the aspect ratio of the bounding box
        float aspectRatio = width / height;

        // Calculate the perimeter of the bounding box
        float perimeter = 2 * (width + height);

        // Combine these factors for a more robust measure
        // You may need to adjust these weights based on your specific use case
        float chestSize = (area * 0.6f) + (aspectRatio * 0.2f) + (perimeter * 0.2f);

        return chestSize;
    }



    private float minChangeThreshold = 0.005f; // Adjust this value as needed

    private void DetectBreathing(float currentChestSize)
    {
        if (lastChestSize == 0f)
        {
            lastChestSize = currentChestSize;
            return;
        }

        float relativeChange = (currentChestSize - lastChestSize) / lastChestSize;

        // Print breathing status
        string breathingStatus;
        if (relativeChange > minChangeThreshold)
        {
            breathingStatus = "Inhale";
        }
        else if (relativeChange < -minChangeThreshold)
        {
            breathingStatus = "Exhale";
        }
        else
        {
            breathingStatus = "No significant change";
        }

        // Log both the numerical data and the breathing status
        Debug.Log($"Current: {currentChestSize}, Last: {lastChestSize}, Relative Change: {relativeChange}, Status: {breathingStatus}");

        // Existing code for significant change detection
        if (relativeChange > breathingThreshold)
        {
            if (!isInhaling)
            {
                isInhaling = true;
                
            }
        }
        else if (relativeChange < -breathingThreshold)
        {
            if (isInhaling)
            {
                isInhaling = false;
                 
            }
        }

        lastChestSize = currentChestSize;
    }

 


    private Texture2D ResizeTexture(Texture2D source, int targetWidth, int targetHeight)
    {
        RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight, 0, RenderTextureFormat.ARGB32);
        Graphics.Blit(source, rt);
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = rt;
        Texture2D result = new Texture2D(targetWidth, targetHeight);
        result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        result.Apply();
        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(rt);
        return result;
    }

    private (Rect box, string label)? GetChestBoundingBox(List<(Rect box, string label)> boundingBoxes)
    {
        foreach (var (box, label) in boundingBoxes)
        {
            if (label == "chest")
            {
                return (box, label);
            }
        }
        return null;
    }

    private float CalculateBBoxSize(Rect bbox)
    {
        return bbox.width * bbox.height;
    }

    private void DrawBoundingBox(float x, float y, float width, float height)
    {
        // Convert YOLO coordinates to screen coordinates
        float screenX = x * Screen.width;
        float screenY = y * Screen.height;
        float screenWidth = width * Screen.width;
        float screenHeight = height * Screen.height;

        // Draw the bounding box (you'll need to create a line renderer or use GUI.DrawTexture in OnGUI)
        // This is a placeholder for where you'd draw the box
        Debug.Log($"Drawing box at: ({screenX}, {screenY}), size: {screenWidth}x{screenHeight}");
    }

    private Rect ConvertYoloToScreenCoordinates(Rect yoloRect)
    {
        return new Rect(
            yoloRect.x * Screen.width,
            yoloRect.y * Screen.height,
            yoloRect.width * Screen.width,
            yoloRect.height * Screen.height
        );
    }

    private List<(Rect box, string label)> ParseYOLOOutput(Tensor output)
    {
        List<(Rect box, string label)> boundingBoxes = new List<(Rect box, string label)>();

        int totalPredictions = output.shape[1];
        int valuesPerPrediction = output.shape[2];

        float[] outputData = output.AsFloats();

        for (int i = 0; i < totalPredictions; i++)
        {
            int offset = i * valuesPerPrediction;
            float confidence = outputData[offset + 4];
            if (confidence > 0.5f)
            {
                float x = outputData[offset + 0];
                float y = outputData[offset + 1];
                float w = outputData[offset + 2];
                float h = outputData[offset + 3];

                float chestProb = outputData[offset + 5];
                float headProb = outputData[offset + 6];

                string label = chestProb > headProb ? "chest" : "head";

                // Convert YOLO format to Unity Rect format
                Rect box = new Rect(x - w / 2, y - h / 2, w, h);
                boundingBoxes.Add((box, label));
            }
        }

        return boundingBoxes;
    }

    void OnApplicationQuit()
    {
        if (worker != null)
        {
            worker.Dispose();
            worker = null;
        }

        if (webCamTexture != null)
        {
            webCamTexture.Stop();
            Destroy(webCamTexture);
            webCamTexture = null;
        }

        if (texture2D != null)
        {
            Destroy(texture2D);
            texture2D = null;
        }
    }
}