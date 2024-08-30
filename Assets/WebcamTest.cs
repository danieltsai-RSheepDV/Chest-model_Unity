using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class WebcamTest : MonoBehaviour
{
    [SerializeField] private RawImage RawImage;
    private WebCamTexture webCamTexture;
    
    // Start is called before the first frame update
    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        webCamTexture = new WebCamTexture(devices[0].name, 640,480, 15);
        webCamTexture.Play();
    }

    // Update is called once per frame
    void Update()
    {
        RawImage.texture = webCamTexture;
    }
}
