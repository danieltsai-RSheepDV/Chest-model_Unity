using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using Unity.Sentis.Layers;
using UnityEngine.Assertions;

public class ModelTester : MonoBehaviour
{
    public ModelAsset modelAsset;
    public RawImage rawImage;

    private Model model;
    private IWorker worker;

    private WebCamTexture webCamTexture;

    // Define constants for the model input dimensions
    private const int MODEL_INPUT_WIDTH = 640;
    private const int MODEL_INPUT_HEIGHT = 640;
    private const int MODEL_INPUT_CHANNELS = 3;

    void Start()
    {
        Application.targetFrameRate = 60;
        
        Debug.Log("Starting initialization...");
        Assert.IsNotNull(modelAsset, "modelAsset is not assigned. Please assign it in the Inspector.");

        try
        {
            model = ModelLoader.Load(modelAsset);
            Assert.IsNotNull(model, "Failed to load model. ModelLoader.Load returned null.");
            Debug.Log("Model loaded successfully.");
            PrintModelInfo();

            worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
            Assert.IsNotNull(worker, "Failed to create worker. WorkerFactory.CreateWorker returned null.");
            Debug.Log("Worker created successfully.");
        
            // Init TensorOps
            allocator = new TensorCachingAllocator();
            ops = WorkerFactory.CreateOps(BackendType.GPUCompute, allocator);
            Assert.IsNotNull(worker, "Failed to create ops. WorkerFactory.CreateOps returned null.");
        }
        catch (Exception e)
        {
            Debug.LogError($"Exception while initializing model or worker: {e.Message}");
            return;
        }

        centersToCorners = new TensorFloat(new TensorShape(4, 4),
            new float[]
            {
                1, 0, 1, 0,
                0, 1, 0, 1,
                -0.5f, 0, 0.5f, 0,
                0, -0.5f, 0, 0.5f
            });

        WebCamDevice[] devices = WebCamTexture.devices;
        Assert.AreNotEqual(devices.Length, 0, "No webcam devices found.");
        Debug.Log($"Number of webcam devices found: {devices.Length}");
        
        for (int i = 0; i < devices.Length; i++)
        {
            Debug.Log($"Webcam {i}: {devices[i].name}");
        }
        
        webCamTexture = new WebCamTexture(devices[0].name, 800, 608);
        
        Assert.IsNotNull(webCamTexture, "Failed to initialize WebCamTexture.");
        
        rawImage.texture = webCamTexture;
        webCamTexture.Play();
        
        Debug.Log("WebCamTexture initialized and started successfully.");
    }

    void Update()
    {
        Assert.IsNotNull(webCamTexture, "webCamTexture is null or not playing");
        Assert.IsTrue(webCamTexture.isPlaying);
        
        string s = WebCamTexture.devices[0].name;
        
        var inputTensor = TextureConverter.ToTensor(webCamTexture, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, MODEL_INPUT_CHANNELS);

        // Execute the model
        worker.Execute(inputTensor);
        
        inputTensor.Dispose();
        
        // Process the output
        var output = worker.PeekOutput() as TensorFloat;
        var (chest, head) = ProcessModelOutput(output);
        float chestX = chest[0],
            chestY = chest[1],
            chestWidth = chest[2],
            chestHeight = chest[3],
            confidence = chest[4];
        ;
        float headX = head[0],
            headY = head[1],
            headWidth = head[2],
            headHeight = head[3]
        ;
        if (confidence > 0.5f) // Adjust this threshold as needed
        {
            float chestSize = (chestWidth / MODEL_INPUT_WIDTH) * (chestHeight / MODEL_INPUT_HEIGHT);
            float headSize = (headWidth / MODEL_INPUT_WIDTH) * (headHeight / MODEL_INPUT_HEIGHT);
            
            DrawBBox(chestX, chestY, chestWidth, chestHeight);
            DetectBreathing(chestSize, headSize);
        }
        else
        {
            Debug.Log("No chest detected with high confidence");
        }
    }

    void OnDisable()
    {
        if (worker != null) worker.Dispose();
        if (ops != null) ops.Dispose();
        if (allocator != null) allocator.Dispose();
        
        centersToCorners.Dispose();
        if (webCamTexture != null)
        {
            webCamTexture.Stop();
            Destroy(webCamTexture);
            webCamTexture = null;
        }
    }

    [SerializeField] private RawImage bbox;
    private void DrawBBox(float x, float y, float w, float h)
    {
        bbox.rectTransform.anchoredPosition = Vector2.Lerp(bbox.rectTransform.anchoredPosition, new Vector2(x, -y), Time.deltaTime);
        bbox.rectTransform.sizeDelta = Vector2.Lerp(bbox.rectTransform.sizeDelta, new Vector2(w, h), Time.deltaTime);
    }
    
    ITensorAllocator allocator;
    Ops ops;
    TensorFloat centersToCorners;
    private readonly int[] start1 = {0, 0, 0};
    private readonly int[] end1 = {1, 4, 8400};
    private readonly int[] start2 = {0, 4, 0};
    private readonly int[] end2 = {1, 7, 8400};
    private readonly int[] transpose = {1, 0};
    private (float[] chest, float[] head) ProcessModelOutput(TensorFloat output)
    {
        float maxConfidence = 0f;
        float x = 0f, y = 0f, width = 0f, height = 0f;

        var boxCorners = ops.Slice(output, start1, end1, null, null) as TensorFloat;
        boxCorners = ops.Reshape(boxCorners, new TensorShape(4, 8400)) as TensorFloat;
        boxCorners = ops.Transpose(boxCorners, transpose) as TensorFloat;
        boxCorners = ops.MatMul(boxCorners, centersToCorners);

        var allScores = ops.Slice(output, start2, end2, null, null) as TensorFloat;
        allScores = ops.Reshape(allScores, new TensorShape(2, 8400)) as TensorFloat;

        var classIDs = ops.ArgMax(allScores, 1, false, false);
        
        boxCorners.MakeReadable();
        classIDs.MakeReadable();
        allScores.MakeReadable();

        float[] chest = new float[5];
        chest[0] = boxCorners[classIDs[0], 0];
        chest[1] = boxCorners[classIDs[0], 1];
        chest[2] = boxCorners[classIDs[0], 2] - chest[0];
        chest[3] = boxCorners[classIDs[0], 3] - chest[1];
        chest[4] = allScores[0, classIDs[0]];

        float[] head = new float[5];
        head[0] = boxCorners[classIDs[1], 0];
        head[1] = boxCorners[classIDs[1], 1];
        head[2] = boxCorners[classIDs[1], 2] - head[0];
        head[3] = boxCorners[classIDs[1], 3] - head[1];
        head[4] = allScores[0, classIDs[1]];
        
        boxCorners.Dispose();
        allScores.Dispose();
        classIDs.Dispose();

        return (chest, head);
    }

    private float minChangeThreshold = 0.0003f; // Adjust this value as needed
    private bool isInhaling = false;
    private float breathingThreshold = 0.01f;
    private float pastAverageChestSize = 0f;
    private List<float> fHistory = new List<float>();
    private int historyLength = 60;

    private bool getAverage(float newSize, out float average)
    {
        average = 0f;
        
        fHistory.Add(newSize);
        if (fHistory.Count < historyLength) return false;
        
        if (fHistory.Count > historyLength) fHistory.RemoveAt(0);
        
        // Sort the list
        List<float> sorted = new List<float>(fHistory);
        sorted.Sort();

        for (int i = 10; i < 50; i++)
        {
            average += sorted[i];
        }
        average /= 40;
        
        return true;
    }

    private void DetectBreathing(float currentChestSize, float currentHeadSize)
    {
        float relativeChestSize = currentChestSize;
        float currentAverageChestSize;

        if (!getAverage(relativeChestSize, out currentAverageChestSize)) return;

        float relativeChange = (currentAverageChestSize - pastAverageChestSize) / pastAverageChestSize;

        // Print breathing status
        string breathingStatus;
        if (relativeChange > minChangeThreshold)
        {
            if (!isInhaling) isInhaling = true;
            breathingStatus = "Inhale";
        }
        else if (relativeChange < -minChangeThreshold)
        {
            if (isInhaling) isInhaling = false;
            breathingStatus = "Exhale";
        }
        else
        {
            breathingStatus = "No significant change";
        }

        // Log both the numerical data and the breathing status
        Debug.Log($"Current: {currentAverageChestSize}, Last: {pastAverageChestSize}, Relative Change: {relativeChange}, Status: {breathingStatus}");
        // Debug.Log($"Status: {breathingStatus}");

        pastAverageChestSize = currentAverageChestSize;
    }

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
}