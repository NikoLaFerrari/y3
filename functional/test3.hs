import Lab3
import Data.List

main :: IO ()
main = do
  putStrLn "=== TESTING LAMBDA CALCULUS FUNCTIONS ==="

  putStrLn "\n-- rename tests --"
  print $ rename id exp2
  print $ rename (const "x") exp2

  putStrLn "\n-- swapvars tests --"
  print $ swapvars ("x","a") exp1
  print $ swapvars ("x","y") exp2

  putStrLn "\n-- alphaEq tests --"
  print $ exp1 == exp1
  print $ exp1 == L "a" (V "a")
  print $ exp1 == L "a" (V "b")
  print $ exp2 == L "a" (L "b" (L "c" (A (A (V "a") (V "c")) (V "b"))))

  putStrLn "\n-- subst tests --"
  print $ subst (exp1, "y") (A (V "y") (V "y"))
  print $ subst (V "d","x") (A (V "x") (L "x" (L "y" (V "x"))))
  print $ subst (L "a" (A (V "y") (V "a")),"x") (L "y" (A (V "x") (V "y"))))

  putStrLn "\n=== STLC type reasoning (just recorded strings) ==="
  putStrLn e1_ty
  putStrLn e2_ty
  putStrLn e3_ty
  putStrLn e4_ty
  putStrLn e5_ty
  putStrLn e6_ty

  putStrLn "\n-- function type tests --"
  print $ fn1 1 True (\a b -> if b then a else 0)
  print $ (fn2 length (\x y -> if x == y then [] else [head x])) [1,2,3]
  print $ fn3 length 7
  print $ fn4 (\f -> f 10)

  putStrLn "\n=== BIDIRECTIONAL TYPE CHECKING TESTS ==="
  print $ check [] (L' "x" (V' "x")) (Fn (TV 0) (TV 0))
  print $ check [] (L' "x" (V' "x")) (Fn (TV 0) (TV 1))
  print $ check [] bcomp (Fn (Fn (TV 0) (TV 0))
                              (Fn (Fn (TV 0) (TV 0))
                              (Fn (TV 0) (TV 0))))
  print $ check [] bcomp (Fn (Fn (TV 1) (TV 2))
                              (Fn (Fn (TV 0) (TV 1))
                              (Fn (TV 0) (TV 2))))
  print $ check [] oneid (Fn (TV 0) (TV 0))

