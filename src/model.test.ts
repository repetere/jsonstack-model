import { FeatureEmbedding, } from './index';

describe('static feature_embedding modules', () => {
  it('should merge two arrays', () => {
    const b = [0, 0, 0, 0];
    const m = [1, 2];
    const b1 = [0, 0, 0, 0];
    const m1 = [];
    const b2 = [0, 0];
    const m2 = [5, 6, 7, 8];
    const merged = FeatureEmbedding.getMergedArray(b, m);
    const merged1 = FeatureEmbedding.getMergedArray(b1, m1);
    const merged2 = FeatureEmbedding.getMergedArray(b2, m2);
    const merged3 = FeatureEmbedding.getMergedArray(b2, m2, false, false);
    expect(merged).toMatchObject([1, 2, 0, 0]);
    expect(merged1).toMatchObject([0, 0, 0, 0]);
    expect(merged2).toMatchObject([5, 6,]);
    expect(merged3).toMatchObject([5, 6, 7, 8]);
  });
});

const corpus = [
  ['prod1','prod2','prod1','prod3'],
  ['prod1','prod2'],
  ['prod3','prod2','prod3'],
  ['prod1','prod2','prod3','prod1'],
];
let FEModel;
let NewFEModel;
let ConvergedFEModel;
let EmbeddingData;
let NewEmbeddingData;
let ConvergedEmbeddingData;

describe('reloading model weights', () => {
  beforeAll(async () => {
    // const randomUni = tf.randomUniform([1, 100], -1, 1);
    // randomUni.print()
    FEModel = new FeatureEmbedding({
      name: 'FEModel',
      embedSize: 5,
      fit: {
        epochs: 30,
        batchSize: 1,
        callbacks: {
          onTrainEnd: (logs: any) => console.log('FEModel onTrainEnd', { logs, }),
          // onEpochBegin: (epoch: number, logs: any) => console.log('FEModel onEpochBegin', { logs, epoch, }),
        }
      },
    });
    await FEModel.train(corpus,[]);
    // const FEweights = await FEModel.predict();
    EmbeddingData = await FEModel.exportEmbeddings();
    // const labeledFEWeights = FEModel.labelWeights(FEweights);
    // console.log({ FEweights, labeledFEWeights });
    // console.log('EmbeddingData', EmbeddingData);
    NewFEModel = new FeatureEmbedding({
      name: 'NewFEModel',
      embedSize: 5,
      fit: {
        epochs: 30,
        batchSize: 1,
        callbacks: {
          onTrainEnd: (logs: any) => console.log('NewFEModel onTrainEnd', { logs, }),
          // onEpochBegin: (epoch: number, logs: any) => console.log('NewFEModel onEpochBegin', { logs, epoch, }),
        }
      },
    });
    await NewFEModel.importEmbeddings(EmbeddingData);
    await NewFEModel.train(EmbeddingData.featureIds,[]);
    NewEmbeddingData = await NewFEModel.exportEmbeddings();    
    ConvergedFEModel = new FeatureEmbedding({
      name: 'ConvergedFEModel',
      embedSize: 5,
      fit: {
        epochs: 50,
        batchSize: 1,
        callbacks: {
          onTrainEnd: (logs: any) => console.log('ConvergedFEModel onTrainEnd', { logs, }),
          // onEpochBegin: (epoch: number, logs: any) => console.log('ConvergedFEModel onEpochBegin', { logs, epoch, }),
        }
      },
    });
    await ConvergedFEModel.train(corpus, []);
    ConvergedEmbeddingData = await ConvergedFEModel.exportEmbeddings();    
  });
  it('should allow for embeddings to be reloaded', async () => {
    const FEModelLoss = FEModel.loss;
    const NewFEModelLoss = NewFEModel.loss;
    const ConvergedFEModelLoss = ConvergedFEModel.loss;
    // const reducedFEWeights = await FEModel.reduceWeights(Object.values(EmbeddingData.labeledWeights));
    // const reducedNewFEWeights = await NewFEModel.reduceWeights(Object.values(NewEmbeddingData.labeledWeights));
    // const reducedConvergedFEWeights = await ConvergedFEModel.reduceWeights(Object.values(ConvergedEmbeddingData.labeledWeights));
    // const FESim = await FEModel.findSimilarFeatures([], { labeledWeights: EmbeddingData.labeledWeights, limit: 2, features: ['prod1'], });
    // const NewFESim = await NewFEModel.findSimilarFeatures([], { labeledWeights: NewEmbeddingData.labeledWeights, limit: 2, features: ['prod1'], });
    // const ConvergedFESim = await ConvergedFEModel.findSimilarFeatures([], { labeledWeights: ConvergedEmbeddingData.labeledWeights, limit: 2, features: ['prod1'], });

    // console.log('FESim',FESim);
    // console.log('NewFESim',NewFESim);
    // console.log('ConvergedFESim',ConvergedFESim);
    // console.log('reducedFEWeights',FEModel.labelWeights(reducedFEWeights));
    // console.log('reducedNewFEWeights', NewFEModel.labelWeights(reducedNewFEWeights));
    // console.log('reducedConvergedFEWeights', ConvergedFEModel.labelWeights(reducedConvergedFEWeights));
    // console.log({ FEModelLoss, NewFEModelLoss, ConvergedFEModelLoss });
    expect(ConvergedFEModelLoss).toBeLessThan(FEModelLoss);
    expect(ConvergedFEModelLoss).toBeLessThan(NewFEModelLoss);
  });
  it('should allow for adding more features', async () => {
    let UpdatedModel = new FeatureEmbedding({
      name: 'UpdatedModel',
      embedSize: 5,
      fit: {
        epochs: 70,
        batchSize: 1,
        callbacks: {
          onTrainEnd: (logs: any) => console.log('ConvergedFEModel onTrainEnd', { logs, }),
          // onEpochBegin: (epoch: number, logs: any) => console.log('ConvergedFEModel onEpochBegin', { logs, epoch, }),
        }
      },
    });
    console.log('EmbeddingData', EmbeddingData);

    await UpdatedModel.importEmbeddings({
      ...EmbeddingData,
      inputMatrixFeatures: [
        ['prod4','prod1','prod4','prod1'],
        ['prod1','prod4','prod4','prod1'],
        ['prod3','prod4','prod5'],
      ]
    });
    // console.log('UpdatedModel.featureToId', UpdatedModel.featureToId);
    // console.log('UpdatedModel.idToFeature', UpdatedModel.idToFeature);
    // console.log('UpdatedModel.featureIds', UpdatedModel.featureIds);
    // console.log('UpdatedModel.numberOfFeatures', UpdatedModel.numberOfFeatures);
    // console.log('UpdatedModel.importedEmbeddings', UpdatedModel.importedEmbeddings);
    await UpdatedModel.train([],[]);
    const UpdatedEmbeddingData = await UpdatedModel.exportEmbeddings();


    const UpdatedModelLoss = UpdatedModel.loss;
    // const ConvergedFEModelLoss = ConvergedFEModel.loss;
    // const reducedConvergedFEWeights = await ConvergedFEModel.reduceWeights(Object.values(ConvergedEmbeddingData.labeledWeights));
    const reducedUpdatedReducedWeights = await UpdatedModel.reduceWeights(Object.values(UpdatedEmbeddingData.labeledWeights));
    // const ConvergedFESim = await ConvergedFEModel.findSimilarFeatures([], { labeledWeights: ConvergedEmbeddingData.labeledWeights, limit: 2, features: ['prod1'], });
    // const UpdatedSim = await UpdatedModel.findSimilarFeatures([], { labeledWeights: UpdatedEmbeddingData.labeledWeights, limit: 2, features: ['prod1'], });

    // console.log('UpdatedSim',UpdatedSim);
    // console.log('ConvergedFESim',ConvergedFESim);
    console.log('reducedUpdatedReducedWeights', UpdatedModel.labelWeights(reducedUpdatedReducedWeights));
    // console.log('reducedConvergedFEWeights', ConvergedFEModel.labelWeights(reducedConvergedFEWeights));
    // console.log({ UpdatedModelLoss, ConvergedFEModelLoss });
    // expect(reducedUpdatedReducedWeights).toHaveProperty('prop5');
    expect(UpdatedModelLoss).toBeLessThan(2);
    const moreData = [
      ['prod5', 'prod2', 'prod7', 'prod2'],
      ['prod2', 'prod5', 'prod7', 'prod2'],
      ['prod6', 'prod5', 'prod5'],
    ];
    await UpdatedModel.importEmbeddings({
      ...UpdatedEmbeddingData,
      inputMatrixFeatures: moreData
    });
    await UpdatedModel.train([],[]);
    const UpdatedEmbeddingData2 = await UpdatedModel.exportEmbeddings();
    const UpdatedModelLoss2 = UpdatedModel.loss;
    // console.log({ UpdatedModelLoss2, UpdatedEmbeddingData2 });
    expect(UpdatedModelLoss2).toBeLessThan(1);


  });
})