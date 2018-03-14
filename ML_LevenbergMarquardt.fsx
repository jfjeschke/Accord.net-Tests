#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\officeAutomata\packages\FSharp.Data.2.4.3\lib\net45\FSharp.Data.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.3.8.0\lib\net45\Accord.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.MachineLearning.3.8.0\lib\net45\Accord.MachineLearning.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.Math.3.8.0\lib\net45\Accord.Math.Core.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.Math.3.8.0\lib\net45\Accord.Math.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.Statistics.3.8.0\lib\net45\Accord.Statistics.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.IO.3.8.0\lib\net45\Accord.IO.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.Neuro.3.8.0\lib\net45\Accord.Neuro.dll"
#r @"C:\Users\Jeremiah Jeschke\Documents\Visual Studio 2015\Projects\Packages\Accord.Genetic.3.8.0\lib\net45\Accord.Genetic.dll"


open System 
open System.Diagnostics
open System.IO
open System.Threading
open System.Net
open System.Collections.Generic
open FSharp.Data
open System.Data
//open Accord.Statistics.Models.Regression
//open Accord.Statistics.Models.Regression.Fitting 
//open Accord.MachineLearning
//open Accord.MachineLearning.VectorMachines
//open Accord.MachineLearning.VectorMachines.Learning
//open Accord.MachineLearning.Bayes
open Accord.Statistics.Filters
//open Accord.Statistics.Kernels
open Accord.Math.Optimization
open Accord.Math.Optimization.Losses
open Accord.Neuro
open Accord.Neuro.Learning
open Accord.Neuro.Networks
open Accord.Genetic
open Accord.IO
open Accord.Math


//This class implements the Levenberg-Marquardt learning algorithm, which treats the neural network learning as a function optimization problem. 
//The Levenberg-Marquardt is one of the fastest and accurate learning algorithms for small to medium sized networks.
//However, in general, the standard LM algorithm does not perform as well on pattern recognition problems as it does on function approximation problems. 
//The LM algorithm is designed for least squares problems that are approximately linear. 
//Because the output neurons in pattern recognition problems are generally saturated, it will not be operating in the linear region.
//The advantages of the LM algorithm decreases as the number of network parameters increases. 

try
    let test_input =
                [|  [|-1.0;1.0|]
                    [|-1.0;1.0|]
                    [|1.0;-1.0|]
                    [|1.0;-1.0|] |]
    let test_output = 
                [|  [|-1.0|]
                    [|1.0|]
                    [|1.0|]
                    [|-1.0|] |]

    let numbers () =
        let inputs =
            [|  [|-1.0; -1.0; -1.0; 0.0|];
                [|-1.0; 1.0; -1.0; 0.0|];
                [|1.0; -1.0; -1.0; 0.0|];
                [|1.0; 1.0; -1.0; 0.0|];
                [|-1.0; -1.0; 1.0; 0.0|];
                [|-1.0; 1.0; 1.0; 0.0|];
                [|1.0; -1.0; 1.0; 0.0|];
                [|1.0; 1.0; 1.0; 1.0|]; |]
        let outputs = 
            [|  0;
                1;
                1;
                0;
                2;
                3;
                3;
                2 |]
        outputs, inputs


    let labels, inputs = numbers()

    let numberOfInputs = 4
    let numberOfClasses = 4
    let hiddenNeurons = 5

    let outputs = Jagged.OneHot(labels, numberOfClasses)

    // Next we can proceed to create our network
    let function0 = new BipolarSigmoidFunction(2.0)
    let network = new ActivationNetwork(function0, numberOfInputs, hiddenNeurons, numberOfClasses);

    // Heuristically randomize the network
    let nn = new NguyenWidrow(network)
    nn.Randomize()

    // Create the learning algorithm
    let teacher = new LevenbergMarquardtLearning(network);

    // Teach the network for 10 iterations:
    let mutable error = Double.PositiveInfinity;
    [|0..10|] |> Array.iter (fun x -> error <- teacher.RunEpoch(inputs, outputs))

    // At this point, the network should be able to 
    // perfectly classify the training input points.
       
    [|0..inputs.Length-1|] |> Array.iter (fun i -> 
       let mutable answer = 0
       let output = network.Compute(inputs.[i])
       let response = output.Max(&answer)
       let expected = labels.[i]
       printfn "%A %A %A" answer response expected)

with
| exn -> printfn "%A" exn

System.Console.ReadKey(true) |>ignore
