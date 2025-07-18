import {
  CANVAS_ID,
  IMAGE_CROPPER_ID,
  RESIZE_IMAGE_CANVAS_ID,
} from "./constants";

export const cropImage = (imagePath, newX, newY, newWidth, newHeight) => {
  return new Promise((resolve, reject) => {
    try {
      // Create an image object from the path
      const originalImage = new Image();

      // Wait for the image to finish loading
      originalImage.addEventListener("load", function () {
        // Initialize the canvas object
        const canvas = document.getElementById(CANVAS_ID);
        if (!canvas) {
          reject(new Error("Canvas element not found"));
          return;
        }
        const ctx = canvas.getContext("2d");

        const imageCropper = document.getElementById(IMAGE_CROPPER_ID);
        if (!imageCropper) {
          reject(new Error("Image cropper element not found"));
          return;
        }

        // Calculate scaling factors after the image has loaded
        const widthScale = originalImage.width / imageCropper.width;
        const heightScale = originalImage.height / imageCropper.height;

        // Set the canvas size to the new width and height
        canvas.width = newWidth * 1;
        canvas.height = newHeight * 1;

        // Draw the image onto the canvas
        ctx.drawImage(
          originalImage,
          newX * widthScale,
          newY * heightScale,
          newWidth * widthScale,
          newHeight * heightScale,
          0,
          0,
          newWidth,
          newHeight
        );

        // Resolve the promise when cropping is complete
        resolve(true);
      });

      originalImage.addEventListener("error", function () {
        reject(new Error("Failed to load image"));
      });

      // Set the image source after setting up event listeners
      originalImage.src = imagePath;
    } catch (e) {
      reject(e);
    }
  });
};

export const getCroppedImg = (setSrc, fileName) => {
  const canvas = document.getElementById(CANVAS_ID);
  const fileExtension = fileName.split(".").pop();
  const dataURL = canvas.toDataURL(`image/png`);
  const newFile = dataURLtoFile(dataURL, fileName);
  setSrc(URL.createObjectURL(newFile));
  return newFile;
};

const dataURLtoFile = (dataurl, filename) => {
  var arr = dataurl.split(","),
    mime = arr[0].match(/:(.*?);/)[1],
    bstr = atob(arr[1]),
    n = bstr.length,
    u8arr = new Uint8Array(n);

  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }

  return new File([u8arr], filename, { type: mime });
};

export const resizeImage = (imagePath, maxWidth) => {
  try {
    //create an image object from the path
    const originalImage = new Image();
    originalImage.src = imagePath;

    //initialize the canvas object
    const canvas = document.getElementById(RESIZE_IMAGE_CANVAS_ID);
    const ctx = canvas.getContext("2d");

    const setWidth =
      originalImage.width <= maxWidth ? originalImage.width : maxWidth;
    const setHeight =
      originalImage.width <= maxWidth
        ? originalImage.height
        : (originalImage.height * maxWidth) / originalImage.width;

    // wait for the image to finish loading
    originalImage.addEventListener("load", function () {
      //set the canvas size to the new width and height
      canvas.width = setWidth;
      canvas.height = setHeight;

      // draw the image
      ctx.drawImage(originalImage, 0, 0, setWidth, setHeight);
    });
    return true;
  } catch (e) {
    return false;
  }
};

export const getResizedImg = (setSrc, fileName) => {
  const canvas = document.getElementById(RESIZE_IMAGE_CANVAS_ID);
  const fileExtension = fileName.split(".").pop();
  const dataURL = canvas.toDataURL(`image/png`);
  const newFile = dataURLtoFile(dataURL, fileName);
  setSrc(URL.createObjectURL(newFile));
  return newFile;
};
