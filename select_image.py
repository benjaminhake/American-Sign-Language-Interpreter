import cv2


labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
input_dir_path = './asl_dataset/asl_alphabet_train/'
output_dir_path = './asl_gray/'

# select 100 images of each sign
# and convert them to grayscale
for lb in labels:
    for i in range(1, 501):
        temp_im = cv2.imread(input_dir_path + lb + '/' + lb + str(6*i) + '.jpg', cv2.IMREAD_COLOR)
        cv2.imwrite(output_dir_path + lb + '/' + lb + str(i-1) + '.png', temp_im)
        # cv2.imwrite(output_dir_path + lb + '/' + lb + str(i-1) + '.png', temp_im)