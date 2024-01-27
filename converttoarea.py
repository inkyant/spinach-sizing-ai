import numpy as np

mask = np.array([[0, 0, 1, 1, 2],
                 [0, 1, 1, 2, 2],
                 [0, 0, 2, 2, 2]])

# Identify unique segment values
unique_segments = np.unique(mask)
segment_sizes = {}

for segment in unique_segments:
    if segment != 0:  #'0' is the background
        segment_sizes[segment] = np.sum(mask == segment)

print("Segment Sizes (in pixels):")
for segment, size in segment_sizes.items():
    print(f"Segment {segment}: {size} pixels")

# **need to identify which segment is the quarter**
    
quarter_key = #segment number of quarter

quarter_pixels = segment_sizes[quarter_key]

scale_factor = 4.622442041/quarter_pixels
pixel_count_quarter = segment_sizes[quarter_key]

for key in segment_sizes:
    segment_sizes[key]*= scale_factor

print("Segment Sizes (in cm):")
for segment, size in segment_sizes.items():
    print(f"Segment {segment}: {size} cm")

#print(segment_sizes)

