import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator

'''
Geometric Correction Factor for Measuring Sheet Resistance with 4 Probe Method
Input: wOverS, lOverW
Output: C (correction factor)

Data From: Haldor Topsøe, Geometric Factors in Four Point Resistivity Measurement, 1966.
'''

rawData = [
    # w/s, l/w, C

    # l/w = 1
    (3,   1, 0.5422),
    (4,   1, 0.6870),
    (5,   1, 0.7744),
    (7.5, 1, 0.8846),
    (10,  1, 0.9313),
    (15,  1, 0.9682),
    (20,  1, 0.9822),
    (40,  1, 0.9955),

    # l/w = 2
    (1.5,  2, 0.3263),
    (1.75, 2, 0.3794),
    (2,    2, 0.4292),
    (2.5,  2, 0.5192),
    (3,    2, 0.5957),
    (4,    2, 0.7115),
    (5,    2, 0.7887),
    (7.5,  2, 0.8905),
    (10,   2, 0.9345),
    (15,   2, 0.9696),
    (20,   2, 0.9830),
    (40,   2, 0.9957),

    # l/w = 3
    (1,    3, 0.2204),
    (1.25, 3, 0.2751),
    (1.5,  3, 0.3286),
    (1.75, 3, 0.3803),
    (2,    3, 0.4297),
    (2.5,  3, 0.5194),
    (3,    3, 0.5958),
    (4,    3, 0.7115),
    (5,    3, 0.7887),
    (7.5,  3, 0.8905),
    (10,   3, 0.9345),
    (15,   3, 0.9696),
    (20,   3, 0.9830),
    (40,   3, 0.9957),

    # l/w = 4
    (1,    4, 0.2205),
    (1.25, 4, 0.2751),
    (1.5,  4, 0.3286),
    (1.75, 4, 0.3803),
    (2,    4, 0.4297),
    (2.5,  4, 0.5194),
    (3,    4, 0.5958),
    (4,    4, 0.7115),
    (5,    4, 0.7887),
    (7.5,  4, 0.8905),
    (10,   4, 0.9345),
    (15,   4, 0.9696),
    (20,   4, 0.9830),
    (40,   4, 0.9957),
]

rawData = np.array(rawData)

wOverS = rawData[:, 0]
lOverW = rawData[:, 1]
C = rawData[:, 2]

points = np.column_stack((wOverS, lOverW))

cubicInterpolator = CloughTocher2DInterpolator(points, C)
linearInterpolator = LinearNDInterpolator(points, C)
nearestInterpolator = NearestNDInterpolator(points, C)


def getCorrectionFactor(wOverSInput, lOverWInput):
    point = np.array([[wOverSInput, lOverWInput]])
    cValue = cubicInterpolator(point)[0]

    if np.isnan(cValue):
        cValue = linearInterpolator(point)[0]

    if np.isnan(cValue):
        cValue = nearestInterpolator(point)[0]
        print("Value is outside the interpolation range. Using nearest neighbor value.")

    return float(cValue)


def getCorrectionGrid(W, L):
    gridPoints = np.column_stack((W.ravel(), L.ravel()))

    cGrid = cubicInterpolator(gridPoints)

    nanMask = np.isnan(cGrid)
    if np.any(nanMask):
        cGrid[nanMask] = linearInterpolator(gridPoints[nanMask])

    nanMask = np.isnan(cGrid)
    if np.any(nanMask):
        cGrid[nanMask] = nearestInterpolator(gridPoints[nanMask])

    return cGrid.reshape(W.shape)


# Grid
wGrid = np.linspace(1, 40, 250)
lGrid = np.linspace(1, 4, 250)
W, L = np.meshgrid(wGrid, lGrid)
CGrid = getCorrectionGrid(W, L)

chosenCmap = "plasma"

# 3D Surface Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(
    W,
    L,
    CGrid,
    cmap=chosenCmap,
    alpha=0.9,
    linewidth=0,
    antialiased=True,
    zorder=2
)

ax.scatter(
    wOverS,
    lOverW,
    C,
    color='gray',
    marker='o',
    s=15,
    linewidths=1,
    label='Data Points',
    zorder=3
)

ax.set_title(
    "3D Interpolation of Rectangular Sample Correction Factor",
    fontsize=13,
    pad=15,
    fontweight='bold'
)

ax.set_xlabel("w / s", fontsize=11, labelpad=10)
ax.set_ylabel("l / w", fontsize=11, labelpad=10)
ax.set_zlabel("Correction Factor, C", fontsize=11, labelpad=10)

ax.view_init(elev=25, azim=-135)

fig.colorbar(
    surface,
    ax=ax,
    shrink=0.65,
    aspect=12,
    pad=0.1,
    label="Correction Factor, C"
)

ax.legend(loc='upper left')

plt.tight_layout()
plt.show()


# Heatmap Top view
fig, ax = plt.subplots(figsize=(9, 6))

heatmap = ax.contourf(
    W,
    L,
    CGrid,
    levels=100,
    cmap=chosenCmap
)

contours = ax.contour(
    W,
    L,
    CGrid,
    levels=10,
    colors='black',
    linewidths=0.6,
    alpha=0.5
)

ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

ax.scatter(
    wOverS,
    lOverW,
    facecolors='none',
    edgecolors='gray',
    marker='o',
    s=30,
    linewidths=1.2,
    label='Data Points'
)

ax.set_title(
    "2D Heatmap of Rectangular Sample Correction Factor",
    fontsize=13,
    pad=15,
    fontweight='bold'
)

ax.set_xlabel("w / s", fontsize=11)
ax.set_ylabel("l / w", fontsize=11)

ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.3)
ax.grid(True, which='minor', color='lightgray', linestyle='--', alpha=0.3)
ax.minorticks_on()

cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label("Correction Factor, C")

ax.legend(loc='upper left')

plt.tight_layout()
plt.show()


# Finding Exact Geometric Correction Factor for Given w/s and l/w
data = input("Enter w/s and l/w values separated by a space: ").split()
wOverSInput, lOverWInput = float(data[0]), float(data[1])
correctionFactor = getCorrectionFactor(wOverSInput, lOverWInput)

print(f"For w/s = {wOverSInput} and l/w = {lOverWInput}, C = {correctionFactor:.4f}")
