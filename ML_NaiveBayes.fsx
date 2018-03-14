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

//A naive Bayes classifier is a simple probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions. 
//A more descriptive term for the underlying probability model would be "independent feature model".
//In simple terms, a naive Bayes classifier assumes that the presence (or absence) of a particular feature of a class is unrelated to the presence (or absence) of any other feature, given the class variable. 
//In spite of their naive design and apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many complex real-world situations.

let datatable2 () = 
    let data = new DataTable("Mitchell's Tennis Example");

    data.Columns.Add("Day") |> ignore
    data.Columns.Add("Outlook") |> ignore
    data.Columns.Add("Temperature") |> ignore
    data.Columns.Add("Humidity") |> ignore
    data.Columns.Add("Wind") |> ignore
    data.Columns.Add("PlayTennis") |> ignore


    data.Rows.Add("D1", "Sunny", "Hot", "High", "Weak", "No") |> ignore
    data.Rows.Add("D2", "Sunny", "Hot", "High", "Strong", "No") |> ignore
    data.Rows.Add("D3", "Overcast", "Hot", "High", "Weak", "Yes") |> ignore
    data.Rows.Add("D4", "Rain", "Mild", "High", "Weak", "Yes") |> ignore
    data.Rows.Add("D5", "Rain", "Cool", "Normal", "Weak", "Yes") |> ignore
    data.Rows.Add("D6", "Rain", "Cool", "Normal", "Strong", "No") |> ignore
    data.Rows.Add("D7", "Overcast", "Cool", "Normal", "Strong", "Yes") |> ignore
    data.Rows.Add("D8", "Sunny", "Mild", "High", "Weak", "No") |> ignore
    data.Rows.Add("D9", "Sunny", "Cool", "Normal", "Weak", "Yes") |> ignore
    data.Rows.Add("D10", "Rain", "Mild", "Normal", "Weak", "Yes") |> ignore
    data.Rows.Add("D11", "Sunny", "Mild", "Normal", "Strong", "Yes") |> ignore
    data.Rows.Add("D12", "Overcast", "Mild", "High", "Strong", "Yes") |> ignore
    data.Rows.Add("D13", "Overcast", "Hot", "Normal", "Weak", "Yes") |> ignore
    data.Rows.Add("D14", "Rain", "Mild", "High", "Strong", "No") |> ignore
    data

let NB_Test () = 
    let table = datatable2()
    let codebook = new Codification(table, "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis")
    let symbols = codebook.Apply(table)

    let ary1 = Matrix.ToJagged(symbols)
    let outputs = ary1.GetColumn(5)  |> Array.map (fun x -> Convert.ToInt32(x))
    let inputs = ary1.GetColumns([|1..4|]) |> Array.map (fun x -> x |> Array.map (fun y -> Convert.ToInt32(y)))
    printfn "%A" outputs
    printfn "%A" inputs

    let learner = new NaiveBayesLearning()

    // and teach a model on the data examples
    let nb = learner.Learn(inputs, outputs)

    let numberOfClasses = nb.NumberOfClasses // should be 2 (positive or negative)
    let nunmberOfInputs = nb.NumberOfInputs
    printfn "%A" numberOfClasses
    printfn "%A" nunmberOfInputs

    let test = [|"Sunny"; "Cool"; "High"; "Strong"|] //"Sunny", "Cool", "High", "Strong"
    let instance = codebook.Transform(test)
    printfn "instance %A" instance

    // Let us obtain the numeric output that represents the answer
    let c = nb.Decide(instance) // answer will be 0
    printfn "c %A" c

    // Now let us convert the numeric output to an actual "Yes" or "No" answer
    let result = codebook.Transform("PlayTennis", "No") // answer will be "No"
    printfn "result %A" result

    // We can also extract the probabilities for each possible answer
    let probs = nb.Probabilities(instance); // { 0.795, 0.205 } //instance
    printfn "probs %A" probs

    // Now, let's test  the model output for the first input sample:
    let answer = nb.Decide([|0; 1; 1; 1|])
    printfn "%A" answer

try
    NB_Test()
with
| exn -> printfn "%A" exn

System.Console.ReadKey(true) |>ignore
