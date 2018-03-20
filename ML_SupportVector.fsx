#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\officeAutomata\packages\FSharp.Data.2.4.3\lib\net45\FSharp.Data.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.3.8.0\lib\net45\Accord.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.MachineLearning.3.8.0\lib\net45\Accord.MachineLearning.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.Math.3.8.0\lib\net45\Accord.Math.Core.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.Math.3.8.0\lib\net45\Accord.Math.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.Statistics.3.8.0\lib\net45\Accord.Statistics.dll"

open System 
open System.Diagnostics
open System.IO
open System.Threading
open System.Net
open System.Collections.Generic
open FSharp.Data
open System.Data
open Accord.Statistics.Models.Regression
open Accord.Statistics.Models.Regression.Fitting 
open Accord.MachineLearning
open Accord.MachineLearning.VectorMachines
open Accord.MachineLearning.VectorMachines.Learning
open Accord.MachineLearning.Bayes
open Accord.Statistics.Filters
open Accord.Statistics.Kernels
open Accord.Math.Optimization
open Accord.Math.Optimization.Losses
open Accord.IO
open Accord.Math


let numbers () =
        let inputs =
            [|  [|1.0; 2.0; |]
                [|1.0; 3.0; |] 
                [|1.0; 0.0; |] 
                [|3.0; 1.0; |]
                [|3.0; 2.0; |] 
                [|3.0; 3.0; |] 
                [|3.0; 0.0; |] 
                [|1.0; 1.0; |]
                [|4.0; 5.0; |] 
                [|4.0; 6.0; |]
                [|4.0; 4.0; |]
                [|0.0; 3.0; |] 
                [|0.0; 1.0; |]
                [|0.0; 0.0; |]
                
                                     |]
        let outputs = 
            [|  0.0;
                0.0;
                0.0;
                1.0;
                1.0;
                1.0;
                1.0;
                0.0;
                2.0;
                2.0;
                2.0;
                0.0;
                0.0;
                0.0;
                |]
        //let o1 = [|0..100|] |> Array.fold (fun acc x -> Array.append outputs acc) outputs   
        //let i1 = [|0..100|] |> Array.fold (fun acc x -> Array.append inputs acc) inputs  
        outputs, inputs
        //o1,i1

let svm_test () =     

    let outputs0, inputs = numbers()  
    let outputs = outputs0 |> Array.map (fun x -> Convert.ToInt32(x))                                                                       
                                                                                                                                        
    let numberofclasses = outputs |> Array.distinct |> fun x -> x.Length

    //Types:
    //Linear, double[] 
    //Gaussian  Hellinger  Polynomial 
    //Sigmoid
    //Multiquadric Laplacian

//___MulticlassSupportVectorLearning
    let teacher = new MulticlassSupportVectorLearning<Gaussian, double[]>() 
    //MulticlassSupportVectorLearning
    //FanChenLinSupportVectorRegression 
    //SequentialMinimalOptimization
    teacher.Learner <- fun x -> let i = new SequentialMinimalOptimization<Gaussian, double[]>()
                                i.UseKernelEstimation <- true //false
                                i.Complexity <- 50.0 //100
                                i.PositiveWeight <- 1.0
                                i.NegativeWeight <- 0.1
                                i.Kernel <- new Gaussian(6.0) //10
                                i.CacheSize <- 0
                                //i.Strategy <- SelectionStrategy.WorstPair 
                                i :> _

    teacher.SubproblemFinished.AddHandler(fun e o -> printfn "%A" o.Progress)               

    // Learn a machine
    let machine = teacher.Learn(inputs, outputs)

    let calibration = new MulticlassSupportVectorLearning<Gaussian, double[]>()
    calibration.Model <- machine
    //BaseSupportVectorCalibration
    //ProbabilisticCoordinateDescent 
    //ProbabilisticOutputCalibration //Linear 
    //StochasticGradientDescent //Linear 
    //SupportVectorReduction //Linear 
    calibration.Learner <- fun x -> let t = new ProbabilisticOutputCalibration<Gaussian, double[]>()
                                    t.Model <- x.Model
                                    t.Iterations <- 150
                                    //t.Tolerance <- 1E5
                                    t :> _

    calibration.SubproblemFinished.AddHandler(fun e o -> printfn "%A" o.Progress)

    let learned = calibration.Learn(inputs, outputs)

    let answers = machine.Decide(inputs) 

    let predicted = machine.Decide(inputs.[1])
    printfn "predicted %A" predicted 

    let probabilities = machine.Probabilities(inputs.[1])
    printfn "Probabilities %A" probabilities 

    let score = machine.Score(inputs.[1])
    printfn "score %A" score 

    let confustionMatrix = new Accord.Statistics.Analysis.GeneralConfusionMatrix(answers, outputs)
    printfn "Accuracy: %f" confustionMatrix.OverallAgreement

    let prob = machine.Probabilities(inputs)
    let error = new ZeroOneLoss(outputs)
    let e0 = error.Loss(answers)
    let loss = new CategoryCrossEntropyLoss(outputs)
    let l0 = loss.Loss(prob)
    printfn "Error: %A" e0
    printfn "Loss: %A" l0

try
    svm_test()
with
| exn -> printfn "%A" exn

System.Console.ReadKey(true) |>ignore
