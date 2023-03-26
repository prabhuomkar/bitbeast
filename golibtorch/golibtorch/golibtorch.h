#ifndef __GOLIBTORCH_H__
#define __GOLIBTORCH_H__

#ifdef __cplusplus

extern "C" {
#endif  // __cplusplus

  typedef void *mModel;
  typedef struct {
    char** labels;
    float* scores;
  } Result;
  mModel NewModel(char *modelFile);
  Result *GetResult(mModel model, float *inputData, int *channels, int *width, int *height);
  void DeleteModel(mModel model);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __GOLIBTORCH_H__