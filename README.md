# Image Augmenter

Python tool for augmentation of object detection images and annotations. Supports annotations in bounding box format (xmin, xmax, ymin, ymax) and as a numpy array of (u, v) point pairs - specified in the construction of the augmenter. Augmenter always returns valid annotations such that all points/bounding boxes are within the augmented image.

Currently has the following augmentation functions implemented (more to come):

* Rotation ```rotate(img, annots, angle)```
* Translation ```translate(img, annots, du, dv)```
* Flip ```flip(img, annots, axis)```
* Zoom ```zoom(img, annots, fx, fy)```
* Blur ```gaussian_blur(img, annots, sigma)```
* Noise Addition ```noise(img, annots, A)```

Two parameters required for constructing ```augmenter```:

* ```randomised```: Choose to randomise the amplitude of the augmentation task. If ```True``` augmenter treats additional parameters (not ```img``` or ```annots```) in the function as the amplitude and will use a random value from that amplitude. If ```False``` will perform augmentation with parameter value provided every time. Good practice to have ```randomised=True``` for large augmentation tasks.
* ```format```: Format of the annotations provided can either be ```"bbox"``` or ```"points"``` based on data used. 

To use augmenter, do something like this:

```
# Augmenter for an array of points described by
# np.array([u1, v1], [u2, v2], ..., [un, vn]])

augmenter = Augmenter(random="True", format="points")
rotated_image = augmenter.rotate(image, annotations, angle)
noisy_image = augmenter.gaussian_blur(image, annotations, amplitude)
```
etc.

