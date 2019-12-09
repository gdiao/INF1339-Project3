console.log('Hello TensorFlow');
/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');  
  const carsData = await carsDataReq.json();  
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
  .filter(car => (car.mpg != null && car.horsepower != null));
  
  return cleaned;
}
//console.log(getData())

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values}, 
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  function createModel() {
      // Create a sequential model
      const model = tf.sequential(); 

      // Add a single hidden layer
      model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
      model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
      model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
      model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
      model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
      // Add an output layer
      model.add(tf.layers.dense({units: 1, useBias: true}));

      return model;
  }
    
  // Create the model
  const model = createModel();  
  tfvis.show.modelSummary({name: 'Model Summary'}, model);
    
  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;
    
  // Train the model  
  await trainModel(model, inputs, labels);
  console.log('Done Training');
    
  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
} // end of RUN


/**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any 
  // intermediate tensors.