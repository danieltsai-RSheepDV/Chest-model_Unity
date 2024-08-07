using UnityEditor;
using Unity.Barracuda;
using UnityEngine;

public class CreateNNModelAsset
{
    [MenuItem("Assets/Create/Barracuda/NNModel")]
    public static void CreateAsset()
    {
        var asset = ScriptableObject.CreateInstance<NNModel>();
        AssetDatabase.CreateAsset(asset, "Assets/NewNNModel.asset");
        AssetDatabase.SaveAssets();
        EditorUtility.FocusProjectWindow();
        Selection.activeObject = asset;
    }
}

