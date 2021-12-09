dur_predict1 = 0
dur_place1 = 0
dur_crop1 = 0
start = timer()
for x in range(int(floor(new_x1/208)/2)):
	for y in range(floor(new_y1/208)):
            start3 = timer()
            img_part1 = cropper(full_image1,y*208,x*208) # overlap = 16 therefore (224 - 16 = 208)
            dur_crop1 = dur_crop1 + timer() - start3
            '''if y==0 and x==1:
                plt.imshow(img_part,vmin = 0,vmax = 255)
                img_part = np.asarray(img_part)
                print(img_part.shape)
            '''
            img_part1 = np.asarray(img_part1)
            img_part1 = img_part1.reshape((1,224,224,3))
            img_part1 = 255 - img_part1
            start1 = timer()
            prediction1 = odmodel.predict(img_part1)
            dur_predict1 = dur_predict1 + timer() - start1
            prediction1 = np.asarray(prediction1,dtype = np.float32)
            #print(prediction.shape)
            print(x," ",y, " 1")
            prediction1 = prediction1.reshape((1,224,224))
            #start1 = timer()
            start2 = timer()
            place(prediction1,full_image_p1,y*208,x*208)
			dur_place1 = dur_place1 + timer()- start2
            #print(timer()-start1)
            #full_image_p[y*224:(y+1)*224,x*224:(x+1)*224] = prediction[0]
            #full_image_p[y*208:y*208 + 224,x*208:x*208 + 224] = prediction[0]
        
duration1 = timer() - start
print(duration1)
print(dur_crop1)
print(dur_predict1)
print(dur_place1)
