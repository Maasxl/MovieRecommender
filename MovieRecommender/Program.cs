using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MovieRecommender
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

            ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

            EvaluateModel(mlContext, testDataView, model);

            UseModelForSinglePrediction(mlContext, model);

            SaveModel(mlContext, trainingDataView.Schema, model);

        }

        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            // Should be replaced with get requests for the prepared ml/ai data
            var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "mockdata.csv");
            var testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "mockdata-test.csv");

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<CampsiteRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<CampsiteRating>(testDataPath, hasHeader: true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "campsiteIdEncoded", inputColumnName: "campsiteId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "campsiteIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");

            var prediction = model.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

        public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
        {
            Console.WriteLine("=============== Making a prediction ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<CampsiteRating, CampsiteRatingPrediciton>(model);

            var testInput = new CampsiteRating { userId = "6", campsiteId = 100006 };

            var ratingPrediction = predictionEngine.Predict(testInput);

            Console.WriteLine("campsite rating for user: " + Math.Round(ratingPrediction.Score, 1));

            if (Math.Round(ratingPrediction.Score, 1) > 3.5)
            {
                Console.WriteLine("Campsite " + testInput.campsiteId + " is recommended for user " + testInput.userId);
            }
            else
            {
                Console.WriteLine("Campsite " + testInput.campsiteId + " is not recommended for user " + testInput.userId);
            }
        }

        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "CampsiteRecommenderModel.zip");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
            Console.WriteLine("Model is saved in " + modelPath);
        }
    }
}
