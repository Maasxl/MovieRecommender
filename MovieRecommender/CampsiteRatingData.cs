using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MovieRecommender
{
    public class CampsiteRating
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float campsiteId;
        [LoadColumn(2)]
        public float Label;
    }

    public class CampsiteRatingPrediciton
    {
        public float Label;
        public float Score;
    }
}
