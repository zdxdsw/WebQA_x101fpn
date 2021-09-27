We've made the following changes to the detectron2 source code:

**detectron2/modeling/meta_arch/rcnn.py &rarr; GeneralizedRCNN**

```
def inference_FE(self, inputs, do_postprocess: bool = True):
        #start = time.time()
        images = self.preprocess_image(inputs)
        
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        outputs, _ = self.roi_heads(images, features, proposals, None)
        #print(f'Time 1: {time.time() - start}')
        for i, instances in enumerate(outputs):
            feature = [features[key][i: i + 1] for key in self.roi_heads.in_features]
            roi_features = self.roi_heads.box_pooler(feature, [instances.pred_boxes])
            head_features, instances.fc1_features = self.roi_heads.box_head.partial_forward(roi_features)
            instances.cls_features = self.roi_heads.box_predictor.cls_score(head_features)
        #print(f'Time 2: {time.time() - start}')
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(outputs, inputs, images.image_sizes)
        else:
            return outputs
        return outputs
```
