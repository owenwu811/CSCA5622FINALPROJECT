#project topic:

#to identify if age or mileage has a bigger affect on depreciation


#project motivation:

#I own a low mileage 2013 boss 302, and I have been scared to drive it, so I hope this project will offer another perspective on my fears for similar cars, confirming or denying my suspicions

#data citations:

#I got car data from carfax listings

#https://www.carfax.com/Used-2007-Ford-Mustang-Shelby-GT500_x7872
#https://www.carfax.com/Used-2008-Ford-Mustang-Shelby-GT500_x9032
#https://www.carfax.com/Used-2010-Ford-Mustang-Shelby-GT500_x11300
#https://www.carfax.com/Used-2011-Ford-Mustang-Shelby-GT500_x12399
#https://www.carfax.com/Used-2012-Ford-Mustang-Shelby-GT500_x13659

#... more


#data size:

#132 s550 gt350 + 134 s197 gt500 (266 cars in total)


#data cleaning:

#remove ford mustang text since we already know that's what it is

#remove "call for price" entries


#we removed transmission from gt350s since all of them have manual tremec only across both standard and Rs

#we removed the used part since we already know these are not in production anymore - s550 gt350s

#we decided to leave the body style for m3s since I had a suspicion that coupes are worth more than verts


#python dictionary to remove duplicate listings - use only keys as tuples and put value as dummy "a"