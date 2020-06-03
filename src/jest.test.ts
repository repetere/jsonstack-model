import { FeatureEmbedding, } from './index';

export function toBeWithinRange(received, floor, ceiling) {
  const pass = received >= floor && received <= ceiling;
  if (pass) {
    return {
      message: () =>
        `expected ${received} not to be within range ${floor} - ${ceiling}`,
      pass: true,
    };
  } else {
    return {
      message: () =>
        `expected ${received} to be within range ${floor} - ${ceiling}`,
      pass: false,
    };
  }
};

expect.extend({
  toBeWithinRange,
});

describe('toBeWithinRage', () => {
  it('numeric ranges', () => {
    expect(100).toBeWithinRange(90, 110);
    expect(101).not.toBeWithinRange(0, 100);
    expect({ apples: 6, bananas: 3 }).toEqual({
      apples: expect.toBeWithinRange(1, 10),
      bananas: expect.not.toBeWithinRange(11, 20),
    });
  });
});

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
    const merged3 = FeatureEmbedding.getMergedArray(b2, m2,false,false);
    expect(merged).toMatchObject([1, 2, 0, 0]);
    expect(merged1).toMatchObject([0, 0, 0, 0]);
    expect(merged2).toMatchObject([5, 6,]);
    expect(merged3).toMatchObject([5, 6,7,8]);
  });
})