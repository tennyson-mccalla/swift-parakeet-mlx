import Foundation
import MLX
import MLXNN

// MARK: - Multi-Head Attention

public class MultiHeadAttention: Module {
    let nHeads: Int
    let nFeat: Int
    let headDim: Int
    let scale: Float

    let linearQ: Linear
    let linearK: Linear
    let linearV: Linear
    let linearOut: Linear

    public init(nHeads: Int, nFeat: Int, bias: Bool = true) {
        self.nHeads = nHeads
        self.nFeat = nFeat
        self.headDim = nFeat / nHeads
        self.scale = 1.0 / sqrt(Float(headDim))

        self.linearQ = Linear(nFeat, nFeat, bias: bias)
        self.linearK = Linear(nFeat, nFeat, bias: bias)
        self.linearV = Linear(nFeat, nFeat, bias: bias)
        self.linearOut = Linear(nFeat, nFeat, bias: bias)

        super.init()
    }

    public func callAsFunction(
        _ q: MLXArray,
        _ k: MLXArray,
        _ v: MLXArray,
        posEmb: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: ConformerCache? = nil
    ) -> MLXArray {
        let q = linearQ(q)
        let k = linearK(k)
        let v = linearV(v)

        let batch = q.shape[0]
        let qSeq = q.shape[1]
        let kSeq = k.shape[1]

        let qReshaped = q.reshaped([batch, qSeq, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        let kReshaped = k.reshaped([batch, kSeq, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        let vReshaped = v.reshaped([batch, kSeq, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])

        // if let cache = cache {
        //     k, v = cache.updateAndFetchConv(k, v)
        // }

        let o = MLX.scaledDotProductAttention(
            queries: qReshaped,
            keys: kReshaped,
            values: vReshaped,
            scale: scale,
            mask: mask
        )

        let output = o.transposed(axes: [0, 2, 1, 3]).reshaped([batch, qSeq, nFeat])

        return linearOut(output)
    }
}

// MARK: - Relative Position Multi-Head Attention

public class RelPositionMultiHeadAttention: MultiHeadAttention {
    var linearPos: Linear
    public var posBiasU: MLXArray
    public var posBiasV: MLXArray

    public init(
        nHeads: Int,
        nFeat: Int,
        bias: Bool = true,
        posBiasU: MLXArray? = nil,
        posBiasV: MLXArray? = nil
    ) {
        // Initialize properties before calling super.init()
        self.linearPos = Linear(nFeat, nFeat, bias: false)

        if let posBiasU = posBiasU {
            self.posBiasU = posBiasU
        } else {
            self.posBiasU = MLXArray.zeros([nHeads, nFeat / nHeads])
        }

        if let posBiasV = posBiasV {
            self.posBiasV = posBiasV
        } else {
            self.posBiasV = MLXArray.zeros([nHeads, nFeat / nHeads])
        }

        super.init(nHeads: nHeads, nFeat: nFeat, bias: bias)
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let H = x.shape[1]
        let Tq = x.shape[2]
        let posLen = x.shape[3]

        // Pad on the last dimension: (0, 0), (0, 0), (0, 0), (1, 0)
        let padded = MLX.padded(
            x,
            widths: [(0, 0), (0, 0), (0, 0), (1, 0)].map { IntOrPair($0) },
            mode: .constant,
            value: MLXArray(0.0)
        )

        let reshaped = padded.reshaped([B, H, posLen + 1, Tq])
        let sliced = reshaped[0..., 0..., 1..., 0...]
        let result = sliced.reshaped([B, H, Tq, posLen])

        return result
    }

    override public func callAsFunction(
        _ q: MLXArray,
        _ k: MLXArray,
        _ v: MLXArray,
        posEmb: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: ConformerCache? = nil
    ) -> MLXArray {
        guard let posEmb = posEmb else {
            fatalError("pos_emb is necessary for RelPositionMultiHeadAttention!")
        }

        let q = linearQ(q)
        let k = linearK(k)
        let v = linearV(v)
        let p = linearPos(posEmb)  // p stands for position

        let batch = q.shape[0]
        let qSeq = q.shape[1]
        let kSeq = k.shape[1]
        let posLen = p.shape[1]

        let qReshaped = q.reshaped([batch, qSeq, nHeads, headDim])
        let qU = (qReshaped + posBiasU).transposed(axes: [0, 2, 1, 3])
        let qV = (qReshaped + posBiasV).transposed(axes: [0, 2, 1, 3])

        let kReshaped = k.reshaped([batch, kSeq, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        let vReshaped = v.reshaped([batch, kSeq, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        let pReshaped = p.reshaped([batch, posLen, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])

        // if let cache = cache {
        //     k, v = cache.update_and_fetch_kv(k, v)
        // }

        let matrixBD = MLX.matmul(qV, pReshaped.swappedAxes(-2, -1))
        let matrixBDShifted = self.relShift(matrixBD)

        // Match Python exactly: matrix_bd[:, :, :, : k.shape[-2]] * self.scale
        // k.shape[-2] is the sequence length dimension
        let kSeqLen = kReshaped.shape[2]  // sequence length dimension

        // Add bounds checking to prevent crash
        let matrixBDLastDim = matrixBDShifted.shape[3]
        guard kSeqLen <= matrixBDLastDim else {
            fatalError(
                "kSeqLen (\(kSeqLen)) > matrixBDLastDim (\(matrixBDLastDim)). Shapes: matrixBDShifted=\(matrixBDShifted.shape), kReshaped=\(kReshaped.shape)"
            )
        }

        let matrixBDScaled = matrixBDShifted[0..., 0..., 0..., 0..<kSeqLen] * scale

        var finalMatrixBD = matrixBDScaled
        if let mask = mask {
            let expandedMask = mask.expandedDimensions(axis: 0)
            finalMatrixBD = MLX.which(expandedMask, MLXArray(-Float.infinity), finalMatrixBD)
        }

        let o = MLX.scaledDotProductAttention(
            queries: qU,
            keys: kReshaped,
            values: vReshaped,
            scale: scale,
            mask: finalMatrixBD
        )

        let output = o.transposed(axes: [0, 2, 1, 3]).reshaped([batch, qSeq, -1])

        return linearOut(output)
    }
}

// MARK: - Local Relative Position Multi-Head Attention

public class RelPositionMultiHeadLocalAttention: RelPositionMultiHeadAttention {
    var contextSize: (Int, Int)

    public init(
        nHeads: Int,
        nFeat: Int,
        bias: Bool = true,
        posBiasU: MLXArray? = nil,
        posBiasV: MLXArray? = nil,
        contextSize: (Int, Int) = (256, 256)
    ) {
        // Initialize contextSize before calling super.init()
        self.contextSize = contextSize

        super.init(
            nHeads: nHeads,
            nFeat: nFeat,
            bias: bias,
            posBiasU: posBiasU,
            posBiasV: posBiasV
        )

        if min(contextSize.0, contextSize.1) <= 0 {
            fatalError("Context size for RelPositionMultiHeadLocalAttention must be > 0.")
        }
    }

    public override func callAsFunction(
        _ q: MLXArray,
        _ k: MLXArray,
        _ v: MLXArray,
        posEmb: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: ConformerCache? = nil
    ) -> MLXArray {
        guard let posEmb = posEmb else {
            fatalError("pos_emb is necessary for RelPositionMultiHeadLocalAttention!")
        }

        let originalQSeq = q.shape[1]

        var actualMask = mask
        if actualMask == nil {
            actualMask = MLXArray.zeros([q.shape[0], q.shape[1]]).asType(.bool)
        }

        let q = linearQ(q)
        let k = linearK(k)
        let v = linearV(v)
        let p = linearPos(posEmb)

        let batch = q.shape[0]
        let qSeq = q.shape[1]
        let kSeq = k.shape[1]
        let posLen = p.shape[1]

        var qReshaped = q.reshaped([batch, qSeq, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        var kReshaped = k.reshaped([batch, kSeq, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        var vReshaped = v.reshaped([batch, kSeq, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])
        let pReshaped = p.reshaped([batch, posLen, nHeads, headDim]).transposed(axes: [0, 2, 1, 3])

        // if let cache = cache {
        //     k, v = cache.update_and_fetch_kv(k, v)
        // }

        // Pad to fit context size
        let w = max(contextSize.0, contextSize.1)
        let padLen = (2 * w - qReshaped.shape[2] % (2 * w)) % (2 * w)

        qReshaped = MLX.padded(
            qReshaped,
            widths: [(0, 0), (0, 0), (0, padLen), (0, 0)].map { IntOrPair($0) },
            mode: .constant,
            value: MLXArray(0.0)
        )

        kReshaped = MLX.padded(
            kReshaped,
            widths: [(0, 0), (0, 0), (0, padLen), (0, 0)].map { IntOrPair($0) },
            mode: .constant,
            value: MLXArray(0.0)
        )

        vReshaped = MLX.padded(
            vReshaped,
            widths: [(0, 0), (0, 0), (0, padLen), (0, 0)].map { IntOrPair($0) },
            mode: .constant,
            value: MLXArray(0.0)
        )

        actualMask = MLX.padded(
            actualMask!,
            widths: [(0, 0), (0, padLen)].map { IntOrPair($0) },
            mode: .constant,
            value: MLXArray(true)
        )

        let qU = qReshaped + posBiasU.expandedDimensions(axis: 1)
        let qV = qReshaped + posBiasV.expandedDimensions(axis: 1)

        let matrixAC = self.matmulQK(qU, kReshaped, w: w)  // (batch, head, seq, 2w + 1)
        let matrixBD = MLX.matmul(qV, pReshaped.swappedAxes(-2, -1))  // (batch, head, seq, 2w + 1)

        // We only add stuff in range and mask off unnecessaries
        // Add bounds checking to prevent index out of range
        let matrixACLastDim = matrixAC.shape[3]
        let matrixBDLastDim = matrixBD.shape[3]
        let leftContextSize = min(contextSize.0, matrixACLastDim, matrixBDLastDim)

        if leftContextSize > 0 {
            matrixAC[0..., 0..., 0..., 0..<leftContextSize] =
                matrixAC[0..., 0..., 0..., 0..<leftContextSize]
                + matrixBD[0..., 0..., 0..., 0..<leftContextSize]
        }

        let rightStartIdx = max(0, 2 * w + 1 - (contextSize.1 + 1))
        let rightContextStart = min(contextSize.0, matrixBDLastDim)

        if rightStartIdx < matrixACLastDim && rightContextStart < matrixBDLastDim {
            matrixAC[0..., 0..., 0..., rightStartIdx...] =
                matrixAC[0..., 0..., 0..., rightStartIdx...]
                + matrixBD[0..., 0..., 0..., rightContextStart...]
        }

        // Add bounds checking for masking operations
        let leftMaskEnd = min(w - contextSize.0, matrixACLastDim)
        let rightMaskStart = min(w + contextSize.1 + 1, matrixACLastDim)

        if leftMaskEnd > 0 {
            matrixAC[0..., 0..., 0..., 0..<leftMaskEnd] = MLXArray(-Float.infinity)
        }
        if rightMaskStart < matrixACLastDim {
            matrixAC[0..., 0..., 0..., rightMaskStart...] = MLXArray(-Float.infinity)
        }

        let scores = matrixAC * scale

        let mask = actualMask!.expandedDimensions(axis: 1).expandedDimensions(axis: -1)
        let floatMask = MLX.which(mask, MLXArray(-Float.infinity), MLXArray(0.0)).asType(
            matrixAC.dtype)
        let ones = MLXArray.ones(like: floatMask)
        let dMask = self.matmulQK(ones, floatMask, w: w)

        let finalScores = scores + dMask

        let attn = MLX.softmax(finalScores, axis: -1)
        let maskedAttn = MLX.which(mask, MLXArray(0.0), attn)
        let out = self.matmulPV(maskedAttn, vReshaped, w: w)

        let reshapedOut = out.reshaped([batch, -1, nHeads * headDim])
        let actualSeqLen = min(originalQSeq, reshapedOut.shape[1])
        let output = reshapedOut[0..., 0..<actualSeqLen]

        return linearOut(output)
    }

    // Metal kernel implementations to match Python version exactly
    private func matmulQK(_ q: MLXArray, _ k: MLXArray, w: Int) -> MLXArray {
        let kernelSource = """
            // D, W are provided as constant
            uint B = q_shape[0];
            uint H = q_shape[1];
            uint S_q = q_shape[2];
            uint S_k = k_shape[2];
            uint K_rel = 2 * W + 1;

            uint target_idx = thread_position_in_grid.x;
            uint k_rel_idx = thread_position_in_grid.y;

            uint s_q_idx = target_idx % S_q;
            uint remaining_idx = target_idx / S_q;
            uint h_idx = remaining_idx % H;
            uint b_idx = remaining_idx / H;
            uint k_offset = uint(int(k_rel_idx));

            uint stick_q_k_idx = S_k - S_q + s_q_idx;
            // stick to right (assuming S_k >= S_q)

            int s_k_idx_signed = int(stick_q_k_idx) + int(k_offset) - int(W);
            bool is_out_of_bounds = (s_k_idx_signed < 0) || (s_k_idx_signed >= S_k);

            float current_sum = 0.0f;

            if (!is_out_of_bounds) {
                uint s_k_idx = uint(s_k_idx_signed);

            
                uint Q_D_stride = D;
                uint Q_S_stride = S_q * Q_D_stride;
                uint Q_H_stride = H * Q_S_stride;
            
                uint K_D_stride = D;
                uint K_S_stride = S_k * K_D_stride;
                uint K_H_stride = H * K_S_stride;

                uint q_base_offset =
                    b_idx * Q_H_stride + h_idx * Q_S_stride + s_q_idx * Q_D_stride;
                uint k_base_offset =
                    b_idx * K_H_stride + h_idx * K_S_stride + s_k_idx * K_D_stride;

                const device T* q_vec_ptr = q + q_base_offset;
                const device T* k_vec_ptr = k + k_base_offset;

                for (uint d_idx = 0; d_idx < D; ++d_idx) {
                    current_sum += (float)(q_vec_ptr[d_idx]) * (float)(k_vec_ptr[d_idx]);
                }
            }

            // out[b, h, s_q, k_rel]
            uint out_idx = target_idx * K_rel + k_rel_idx;
            if (is_out_of_bounds) {
                out[out_idx] = -INFINITY;
            } else {
                out[out_idx] = (T) current_sum;
            }
            """

        let B = q.shape[0]
        let H = q.shape[1]
        let S_q = q.shape[2]
        let D = q.shape[3]
        // S_k is accessed within the kernel from k_shape[2]

        let outputShape = [B, H, S_q, 2 * w + 1]

        let gridDimX = B * H * S_q
        let gridDimY = 2 * w + 1
        let gridDimZ = 1

        let tgY = min(gridDimY, 32)
        let tgX = min(gridDimX, 1024 / tgY)

        let kernelFn = MLX.MLXFast.metalKernel(
            name: "local_qk_matmul",
            inputNames: ["q", "k"],
            outputNames: ["out"],
            source: kernelSource
        )

        let outputs = kernelFn(
            [q, k],
            template: [
                ("T", q.dtype),
                ("W", w),
                ("D", D),
            ],
            grid: (max(1, gridDimX), max(1, gridDimY), gridDimZ),
            threadGroup: (max(1, tgX), max(1, tgY), 1),
            outputShapes: [outputShape],
            outputDTypes: [q.dtype]
        )

        return outputs[0]
    }

    private func matmulPV(_ prob: MLXArray, _ v: MLXArray, w: Int) -> MLXArray {
        let kernelSource = """
            // D, W, D_v are provided as constant
            uint B = prob_shape[0];
            uint H = prob_shape[1];
            uint S_p = prob_shape[2];
            uint S_v = v_shape[2];
            uint K_rel = 2 * W + 1;

            uint d_idx = thread_position_in_grid.x;
            uint s_p_idx = thread_position_in_grid.y;
            uint bh_idx = thread_position_in_grid.z;  // merged

            if (d_idx >= D_v || s_p_idx >= S_p || bh_idx >= (B * H)) {
                return;
            }

            uint b_idx = bh_idx / H;
            uint h_idx = bh_idx % H;

            float current_sum = 0.0f;

            // p[b, h, s_p, k_rel]
            uint P_H_stride = S_p * K_rel;
            uint P_B_stride = H * P_H_stride;

            // v[b, h, s_v, d]
            uint V_H_stride = S_v * D_v;
            uint V_B_stride = H * V_H_stride;

            // out[b, s_p, h, d]
            uint O_S_stride = D_v * H;
            uint O_B_stride = S_p * O_S_stride;
            
            uint stick_p_v_idx = S_v - S_p + s_p_idx;
            // stick to right (assuming S_v >= S_p)
            
            for (uint k = 0; k < K_rel; ++k) {
                int s_v_idx_signed = int(stick_p_v_idx) + int(k) - int(W);  // for boundary check
                if (s_v_idx_signed >= 0 && s_v_idx_signed < S_v) {
                    uint s_v_idx = uint(s_v_idx_signed);
                    uint prob_idx =
                        b_idx * P_B_stride + h_idx * P_H_stride + s_p_idx * K_rel + k;
                    uint v_idx =
                        b_idx * V_B_stride + h_idx * V_H_stride + s_v_idx * D_v + d_idx;
                    current_sum += prob[prob_idx] * v[v_idx];
                }
            }

            uint out_idx =
                b_idx * O_B_stride + s_p_idx * O_S_stride + h_idx * D_v + d_idx;

            context_out[out_idx] = current_sum;
            """

        let B = prob.shape[0]
        let H = prob.shape[1]
        let S_p = prob.shape[2]
        let K_rel = prob.shape[3]
        // S_v is accessed within the kernel from v_shape[2]
        let D_v = v.shape[3]

        let outputShape = [B, S_p, H, D_v]

        let gridDimX = D_v
        let gridDimY = S_p
        let gridDimZ = B * H

        let tgX = min(gridDimX, 32)
        let tgY = min(gridDimY, 1024 / tgX)

        let kernelFn = MLX.MLXFast.metalKernel(
            name: "local_pv_matmul",
            inputNames: ["prob", "v"],
            outputNames: ["context_out"],
            source: kernelSource
        )

        let outputs = kernelFn(
            [prob, v],
            template: [
                ("T", prob.dtype),
                ("W", w),
                ("D", K_rel),
                ("D_v", D_v),
            ],
            grid: (gridDimX, gridDimY, gridDimZ),
            threadGroup: (max(1, tgX), max(1, tgY), 1),
            outputShapes: [outputShape],
            outputDTypes: [prob.dtype]
        )

        return outputs[0]
    }
}
