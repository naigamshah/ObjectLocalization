dur_predict2 = 0
dur_place2 = 0
dur_crop2 = 0
start = timer()
for x in range(int(floor(new_x2/208)/2),floor(new_x2/208)):
	for y in range(floor(new_y2/208)):
            start3 = timer()
            img_part2 = cropper(full_image2,y*208,x*208) # overlap = 16 therefore (224 - 16 = 208)
            dur_crop2 = dur_crop2 + timer() - start3
            '''if y==0 and x==1:
                plt.imshow(img_part,vmin = 0,vmax = 255)
                img_part = np.asarray(img_part)
                print(img_part.shape)
            '''
            img_part2 = np.asarray(img_part2)
            img_part2 = img_part2.reshape((1,224,224,3))
            img_part2 = 255 - img_part2
            start1 = timer()
            prediction2 = odmodel.predict(img_part2)
            dur_predict2 = dur_predict2 + timer() - start1
            prediction2 = np.asarray(prediction2,dtype = np.float32)
            #print(prediction.shape)
            print(x," ",y," 2")
            prediction2 = prediction2.reshape((1,224,224))
            #start1 = timer()
            start2 = timer()
            place(prediction2,full_image_p2,y*208,x*208)
            dur_place2 = dur_place2 + timer()- start2
            #print(timer()-start1)
            #full_image_p[y*224:(y+1)*224,x*224:(x+1)*224] = prediction[0]
            #full_image_p[y*208:y*208 + 224,x*208:x*208 + 224] = prediction[0]

duration2 = timer() - start
print(duration2)
print(dur_crop2)
print(dur_predict2)
print(dur_place2)
