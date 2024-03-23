use std::path::Path;
use tract_onnx::prelude::*;

fn main() {
    let model = tract_onnx::onnx()
        .model_for_path(Path::new("./such-toxic.onnx"))
        .unwrap()
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap();

    let input_tensor: Tensor =
        tract_ndarray::Array2::from_shape_vec((1, 1536), vec![0.0 as f32; 1536])
            .unwrap()
            .into();

    let outputs = model.run(tvec!(input_tensor.into()));
    println!("{:?}", outputs)
}
