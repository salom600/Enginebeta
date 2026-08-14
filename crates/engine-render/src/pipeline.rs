//! Opaque handle to a pipeline registered with the renderer. (Reserved for
//! future multi-pipeline support; today there's exactly one pipeline.)

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct RenderPipelineHandle(pub u32);
