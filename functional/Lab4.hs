{-# OPTIONS_GHC -Wno-noncanonical-monad-instances #-}
import Control.Monad.State
import Control.Monad.Fail
import System.Random
import Data.List

data Expr = Con Double | Sub Expr Expr | Div Expr Expr
    deriving (Show,Eq)

e1 = Sub (Div (Con 2) (Con 4)) (Con 3)
e2 = Sub (Con 1) (Div (Con 2) (Con 2))
e3 = Div (Con 1) (Sub (Con 2) (Con 2))

-- Exercise 1a
evalSafe :: Expr -> Maybe Double
evalSafe (Con c) = Just c
evalSafe (Sub a b) = (-) <$> evalSafe a <*> evalSafe b
evalSafe (Div a b) = do 
  x <- evalSafe a
  y <- evalSafe b
  if y /= 0 then Just (x / y) else Nothing

-- Exercise 1b
evalSafeMF :: MonadFail m => Expr -> m Double
evalSafeMF (Con c) = return c 
evalSafeMF (Sub a b) = do x <- evalSafeMF a; y <- evalSafeMF b; return (x - y)
evalSafeMF (Div a b) = do 
  x <- evalSafeMF a 
  y <- evalSafeMF b 
  if y /= 0 then return (x / y) else fail "division by 0"

{- different outputs of evalSafeMF ... -}

evalWeird :: Expr -> StateT Int Maybe Double
evalWeird (Con c)    =
  get >>= \n ->
  put (n+1) >>= \_ ->
  return (if n `mod` 3 == 2 then 0 else c)
evalWeird (Sub e1 e2) =
  evalWeird e1 >>= \x1 ->
  evalWeird e2 >>= \x2 ->
  return (x1-x2)
evalWeird (Div e1 e2) =
  evalWeird e1 >>= \x1 ->
  evalWeird e2 >>= \x2 ->
  if x2 /= 0 then return (x1/x2) else lift Nothing
evalWeirdTop e = runStateT (evalWeird e) 0 >>= \(x,s) -> return x

-- Exercise 1c
evalWeird' :: MonadFail m => Expr -> StateT Int m Double
evalWeird' (Con c) = do 
  n <- get 
  put (n+1)
  return (if n `mod` 3 == 2 then 0 else c) 
evalWeird' (Sub a b) = do 
  x <- evalWeird' a 
  y <- evalWeird' b 
  return (x - y)
evalWeird' (Div a b) = do 
  x <- evalWeird' a 
  y <- evalWeird' b 
  if y /= 0 then return (x / y) else lift (fail "division by 0")

evalWeirdTop' :: MonadFail m => Expr -> m Double
evalWeirdTop' e = evalSafeMF (evalWeird' e) 0

data Bin a = L a | B (Bin a) (Bin a)
  deriving (Show,Eq)

mapBin :: (a -> b) -> Bin a -> Bin b
mapBin f (L x)     = L (f x)
mapBin f (B tL tR) = B (mapBin f tL) (mapBin f tR)

instance Functor Bin where
  fmap = mapBin

-- Exercise 2a
{- 
Functor laws hold:
fmap id (L x) = L (id x) = L x
fmap id (B l r) = B (fmap id l) (fmap id r) = B l r

fmap (g . h) (L x) = L (g (h x)) = fmap g (L (h x)) = (fmap g . fmap h) (L x)
fmap (g . h) (B l r) = B (..) (..) distributes similarly by induction.
-}

-- Exercise 2b
instance Monad Bin where
  return = L 
  (L x) >>= f = f x 
  (B l r) >>= f = B (l >>= f) (r >>= f)

instance Applicative Bin where
  pure = return
  fm <*> xm = fm >>= \f -> xm >>= return . f

-- Exercise 2c (optional)
{-
Left to right tree-structure substitution preserves monad laws by structural induction.
-}

-- Exercise 2d (optional)
{- 
This monad replaces each leaf with the tree produced by f, preserving branching.
-}

class Monad m => SelectMonad m where
  select :: [a] -> m a

instance SelectMonad [] where
  select = id

instance SelectMonad IO where
  select xs
    | not (null xs) = do i <- getStdRandom (randomR (0, length xs-1))
                         return (xs !! i)
    | otherwise     = fail "cannot select from empty list"

newtype Dist a = Dist { dist :: [(a,Rational)] }  deriving (Show)

instance Monad Dist where
  return x = Dist [(x,1)]
  xm >>= f = Dist [(y,p*q) | (x,p) <- dist xm, (y,q) <- dist (f x)]

instance SelectMonad Dist where
  select xs
    | not (null xs) = let n = length xs in Dist [(x, 1 / fromIntegral n) | x <- xs]
    | otherwise     = error "cannot select uniformly from an empty list"

instance Functor Dist where
  fmap f xm = xm >>= return . f

instance Applicative Dist where
  pure = return
  xm <*> ym = xm >>= \x -> ym >>= return . x

experiment :: SelectMonad m => m Int
experiment = do
  x <- select [1..6]
  y <- select [1..x]
  return (x + y)

prob :: Eq a => Dist a -> a -> Rational
prob xm x = sum [p | (y,p) <- dist xm, x == y]

normalize :: Eq a => Dist a -> Dist a
normalize xm = Dist [(x,prob xm x) | x <- support xm]
  where
    support :: Eq a => Dist a -> [a]
    support xm = nub [x | (x,p) <- dist xm, p > 0]  -- "nub", defined in Data.List, removes duplicates

-- Exercise 3a
coin :: SelectMonad m => m Bool
coin = select [False, True]

-- Exercise 3b
subset :: SelectMonad m => [a] -> m [a]
subset [] = return []
subset (x:xs) = do
  b <- coin
  ys <- subset xs
  return (if b then x:ys else ys)

-- Exercise 3c
simulate :: Monad m => Int -> m Bool -> m Int
simulate 0 _ = return 0
simulate n gen = do
  b <- gen
  k <- simulate (n-1) gen
  return (if b then k+1 else k)

-- Exercise 3d (optional)
genTree :: SelectMonad m => [a] -> m (Bin a)
genTree = undefined

