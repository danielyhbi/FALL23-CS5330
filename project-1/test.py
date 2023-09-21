def formatRGBValue(number: float) -> int:
    if number < 0:
        return 0
    
    if (number > 255):
        return 255

    return int(number)


def matrixCon(src, dst, kernel):
    channels = [0, 1, 2]

    for row in range(0, len(dst)):

        for col in range(0, len(dst[0])):

            #for channel in range(0, 3):
                rowCenter = int(len(kernel) / 2)
                colCenter = int(len(kernel[0]) / 2)

                sum = 0

                #iterate through the kernel
                for kernalRow in range(-rowCenter, rowCenter+1):
                    for kernalCol in range(-colCenter, colCenter+1):
                
                        currentConvRow = row + kernalRow
                        currentConvCol = col + kernalCol

                        # check if index is out of bounds
                        if (currentConvRow < 0 or currentConvRow >= len(src) or currentConvCol < 0 or currentConvCol >= len(src[0])):
                            continue
                        

                        valFromSrc = src[currentConvRow][currentConvCol]
                        valFromKernel = kernel[kernalRow + rowCenter][kernalCol + colCenter]

                        sum += valFromSrc * valFromKernel

                dst[row][col] = formatRGBValue(sum)

src = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

dst = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

kernel = [[-1], [0], [1]]

print(dst)
matrixCon(src, dst, kernel)
print(dst)

