//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  Chapter:      F2.1 Interpreting an Image: Classification
//  Checkpoint:   F21c
//  Author:       Andréa Puzzi Nicolau, Karen Dyson, David Saah, Nicholas Clinton
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// New section
// Create an Earth Engine Point object over Milan.
var pt = ee.Geometry.Point([9.453, 45.424]);

// Filter the Landsat 8 collection and select the least cloudy image.
var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(pt)
    .filterDate('2019-01-01', '2020-01-01')
    .sort('CLOUD_COVER')
    .first();

// Center the map on that image.
Map.centerObject(landsat, 8);

//  -----------------------------------------------------------------------
//  CHECKPOINT
//  -----------------------------------------------------------------------

// Combine training feature collections.
var trainingFeatures = ee.FeatureCollection([
    forest, developed, water, herbaceous
]).flatten();

// Add NDVI
// Extract the near infrared and red bands.
var nir = landsat.select('SR_B5');
var red = landsat.select('SR_B4');

// Calculate the numerator and the denominator using subtraction and addition respectively.
var numerator = nir.subtract(red);
var denominator = nir.add(red);

// Now calculate NDVI.
var NDVI = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('ndvi')

// Add the layer to our map with a palette.
var vegPalette = ['red', 'white', 'green'];
Map.addLayer(NDVI, {
    min: -1,
    max: 1,
    palette: vegPalette
}, 'NDVI Manual');

landsat = landsat.addBands(NDVI)

// Add Landsat image to the map.
var visParams = {
    bands: ['SR_B4', 'SR_B3', 'SR_B2'],
    min: 7000,
    max: 12000
};
Map.addLayer(landsat, visParams, 'Landsat 8 image');

// Define prediction bands.
var predictionBands = [
    'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
    'ST_B10'
];

// Sample training points.
var classifierTraining = landsat.select(predictionBands)
    .sampleRegions({
        collection: trainingFeatures,
        properties: ['class'],
        scale: 30
    });

//////////////// CART Classifier ///////////////////

// Train a CART Classifier.
var classifier = ee.Classifier.smileCart().train({
    features: classifierTraining,
    classProperty: 'class',
    inputProperties: predictionBands
});

// Classify the Landsat image.
var classified = landsat.select(predictionBands).classify(classifier);

// Define classification image visualization parameters.
var classificationVis = {
    min: 0,
    max: 3,
    palette: ['589400', 'ff0000', '1a11ff', 'd0741e']
};

// Add the classified image to the map.
Map.addLayer(classified, classificationVis, 'CART classified');

/////////////// Random Forest Classifier /////////////////////

// Train RF classifier.
var RFclassifier = ee.Classifier.smileRandomForest(50).train({
    features: classifierTraining,
    classProperty: 'class',
    inputProperties: predictionBands
});

// Classify Landsat image.
var RFclassified = landsat.select(predictionBands).classify(
    RFclassifier);

// Add classified image to the map.
Map.addLayer(RFclassified, classificationVis, 'RF classified');
print(classifierTraining)

// GRAD SECTION
// Calculate NDVI and NDWI
var NDVI = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
var NDWI = landsat.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
// Add bands to image
landsat = landsat.addBands([NDVI, NDWI])
// Define prediction bands to include NDVI.
var predictionBands = [
    'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
    'ST_B10', 'NDVI', 'NDWI'
];
// Sample training points.
var classifierTraining = landsat.select(predictionBands)
    .sampleRegions({
        collection: trainingFeatures,
        properties: ['class'],
        scale: 30
    });
// Train RF classifier.
var RFclassifier = ee.Classifier.smileRandomForest(50).train({
    features: classifierTraining,
    classProperty: 'class',
    inputProperties: predictionBands
});

// Classify Landsat image.
var RFclassified = landsat.select(predictionBands).classify(
    RFclassifier);

// Add classified image to the map.
Map.addLayer(RFclassified, classificationVis, 'RF with NDVI and NDWI');
