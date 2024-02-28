# Form Analysis

## Method:

We take an algorithmic approach to caclulate a score for a rep of a certain exercise.
Given a time series of 3D keypoints, we calcuate angle between joints, velocity, and distance between joints.
To calculate the score, we compare these values to a predefined range of 'perfect' values for a specific exercise

## Skeleton Keypoints:

![H36M](../assets/H36M.png)

_We use the Human3.6m 17 keypoints_

| Joint Number | Joint Name     |
| ------------ | -------------- |
| 0            | Root           |
| 1            | Right Hip      |
| 2            | Right Knee     |
| 3            | Right Ankle    |
| 4            | Left Hip       |
| 5            | Left Knee      |
| 6            | Left Ankle     |
| 7            | Belly          |
| 8            | Neck           |
| 9            | Nose           |
| 10           | Head           |
| 11           | Left Shoulder  |
| 12           | Left Elbow     |
| 13           | Left Wrist     |
| 14           | Right Shoulder |
| 15           | Right Elbow    |
| 16           | Right Wrist    |

## Calculations:

| Metric   | Formula                                                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Angle    | $\theta = \cos^{-1}\left(\frac{{\mathbf{a} \cdot \mathbf{b}}}{{\lVert \mathbf{a} \rVert \cdot \lVert \mathbf{b} \rVert}}\right)$                              |
| Velocity | $\mathbf{v}_{P_{F_2/F_1}} = \frac{d\mathbf{r}_2}{dt} = \frac{d(\mathbf{r}_1 + \mathbf{r}_{1P})}{dt} = \frac{d\mathbf{r}_1}{dt} + \frac{d\mathbf{r}_{1P}}{dt}$ |
| Distance | $d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}$                                                                                                    |
