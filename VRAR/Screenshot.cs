using UnityEngine;

public class Screenshot : MonoBehaviour
{
	
	public void Update()
	{
		if (Input.GetKeyUp(KeyCode.J))
		{
			ScreenCapture.CaptureScreenshot("Screenshot.png");
		}
	}
}