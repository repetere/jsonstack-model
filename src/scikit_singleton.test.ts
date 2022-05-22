import { getScikit, setScikit, } from "./scikitjs_singleton";

describe('tensorflow singleton',()=>{
  describe('getScikit',()=>{
    it('should throw an error if tensorflow has not been set',()=>{
      expect(getScikit.bind(null)).toThrowError(/Looks like you are/);
    });
  });
  describe('setScikit',()=>{
    const scikitMockBackend = {};
    it('should set a scikit singleton',()=>{
      setScikit(scikitMockBackend);
      expect(getScikit()).toBe(scikitMockBackend);
    });
  });
});