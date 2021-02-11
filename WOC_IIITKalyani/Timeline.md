#  Timeline for WOC

# Week 1

--> Decide the structure of application

--> Figure out which algorithm to use 

--> Learn using python libraries(numpy, matplotlib, OpenCv ) , building api's with python  


# Week 2

--> Start implementing often used numpy, matplotlib , Opencv syntaxes 

--> Research more on color correction for color-blindness

--> Built API for hard code of color correction filter (on whole frame {realtime})

# Week 3

--> Clean up the hard code to mask **seletively** only on the wavelengths of color which color-blind struggles to discriminate between

--> Code a color-blind simulator to feed frames created by the simulator to the hard code model to compare original image vs corrected image seen by color-blind
  
  (If Original image **is same as the** corrected image(outputed from simulated image) means it is not working as it is necessary to cut off chunks of wavelength of LMS colorspace which overlap each other )
  
 # Week 4
 
 --> Convert the hard code into a working ML model
 
 # Week 5
 
 --> Improve real time filtering
 
 --> Train the model with more data
 
 --> Make changes to code depending on feedback from a colorblind
