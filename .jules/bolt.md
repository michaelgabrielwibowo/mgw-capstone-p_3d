## 2024-05-22 - Meshgrid Caching in Loop
**Learning:** Repeatedly creating `np.meshgrid` in a high-frequency loop (like video processing) is a significant bottleneck.
**Action:** Always look for meshgrid creation inside `process_frame` or similar loops and move it to initialization or cache it if dimensions are constant.
