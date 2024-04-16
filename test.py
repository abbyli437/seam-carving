# for (int row_start = 1; row_start + 3 < rowsize; row_start += 4) {
#     #pragma omp parallel for private(a, b, c)
#     for (int top_triangle = 0; top_triangle < (colsize + 5) / 6; top_triangle++) {
#         for (int row = row_start; row < row_start + 3; row++) {
#             int first = top_triangle * 6 + row - row_start;
#             int last = (top_triangle + 1) * 6 - row + row_start;
#             for (int col = first; col < min(last, colsize); col++) {
#                 a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
#                 b = cumulative_energy_map.at<double>(row - 1, col);
#                 c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

#                 cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
#             }
#         }
#     }
# }

rowsize = 8
colsize = 11
for row_start in range(1, rowsize - 3, 4):
    for top_triangle in range(0, (colsize + 5) // 6):
        for row in range(row_start, row_start + 3):
            first = top_triangle * 6 + row - row_start
            last = (top_triangle + 1) * 6 - row + row_start
            for col in range(first, min(last, colsize)):
                print(row,col)
print()
# for (int row_start = 1; row_start + 3 < rowsize; row_start += 4) {
#     #pragma omp parallel for private(a, b, c)
#     for (int bottom_triangle = 0; (bottom_triangle + 8) / 6; bottom_triangle++) {
#         for (int row = row_start + 1; row < row_start + 4; row++) {
#             int first = bottom_triangle * 6 - (row - row_start)
#             int last = bottom_triangle * 6 + (row - row_start)
#             for (int col = max(0, first); col < min(last, colsize); col++) {
#                 a = cumulative_energy_map.at<double>(row - 1, max(col - 1, 0));
#                 b = cumulative_energy_map.at<double>(row - 1, col);
#                 c = cumulative_energy_map.at<double>(row - 1, min(col + 1, colsize - 1));

#                 cumulative_energy_map.at<double>(row, col) = energy_image.at<double>(row, col) + std::min(a, min(b, c));
#             }
#         }
#     }
# }

for row_start in range(1, rowsize - 3, 4):
    for bottom_triangle in range(0, (colsize + 8) // 6):
        for row in range(row_start + 1, row_start + 4):
            first = bottom_triangle * 6 - (row - row_start)
            last = bottom_triangle * 6 + (row - row_start)
            for col in range(max(0, first), min(last, colsize)):
                print(row,col)

