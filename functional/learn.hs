import Data.List
import System.IO

sumof = sum [1..1000]

addEx = 5 + 4
subEx = 5 - 4
mulEx = 5 * 4
divEx = 5 / 4
modEx = mod 5 4 -- prefix - because it is before the numbers
modEx2 = 5 `mod` 4 -- infix - because it is inbetween the two numbers

negNumEx  = 5 + (-4)
num9 = 9 :: Int
sqrtOf9 = sqrt (fromIntegral num9) -- formIntegral: Int -> Float

tandf = True && False
torf = True || False


-- List operations:
primeN = [3,5,7,11]
moreP = primeN ++ [13,17,19,23,29]
favN = 2 : 7 : 21 : 66 :[] -- to add numbers to a list x:[]
multListy = [[3,5,7], [11,13,17]]
moreP2 = 2:moreP
revPrime = reverse moreP2
secondP = moreP2 !! 1
firstP = head moreP2
lastP = last moreP2
primeInit = init moreP2 -- init: returns List without last elem
first3P =  take 3 moreP2 -- take: returns first n elem in List
remP = drop 3 moreP2 -- drop: returns List without first n elem
maxP = maximum moreP2
minP = minimum moreP2
sumP = sum moreP2

newL = [2,3,5]
prodP = product newL

ztt = [0..10]
evenL = [2,4..20]
letterL = ['A','C'..'Z']
infL = [10,20..]
