Shader "CW/DistortionCorrectionShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
		[MaterialToggle] _pincushionCorrection ("Correct Pincushion", Float) = 0
		_c1 ("C1 (Frag only)", Float) = -100
		_c2 ("C2 (Frag only)", Float) = 10
		[MaterialToggle] _lcaCorrection ("Correct LCA", Float) = 0
		[MaterialToggle] _meshPincushionCorrection ("Correct Pincusion via Mesh", Float) = 0
		[MaterialToggle] _meshPincushionCorrectionUndo ("Invert Correct Pincusion via Mesh", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

			float _c1;
			float _c2;
			float _lcaCorrection;
			float _pincushionCorrection;
			float _meshPincushionCorrection;
			float _meshPincushionCorrectionUndo;
            sampler2D _MainTex;
            float4 _MainTex_ST;
			
			float distort (float c1, float c2, float r)
			{
				float f = r + c1 * pow(r, 3) + c2 * pow(r, 5);
				float f_prime = c1 * pow(f, 2) + c2 * pow(f, 4) + pow(c1, 2) * pow(f, 4) + pow(c2, 2) * pow(f, 8) + 2 * c1 * c2 * pow(f, 6);
				f_prime = f_prime  / (1 + 4 * c1 * pow(f, 2) + 6 * c2 * pow(f, 4));
				return f_prime;
			}

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
				if(_meshPincushionCorrection == 1) {
					float2 h = o.vertex.xy / o.vertex.w;
             
					float r = sqrt(h.x * h.x + h.y * h.y);
					float f_prime = distort(-0.252, 0.07, r);
	 
					h = h + h * f_prime;
	 
					o.vertex.xy = h.xy * o.vertex.w;
					if(_meshPincushionCorrectionUndo == 1) {
						float2 h2 = o.vertex.xy / o.vertex.w;
             
						float r2 = sqrt(h2.x * h2.x + h2.y * h2.y);
						float f_prime2 = distort(0.4, 0.62, r2);
	 
						h2 = h2 + h2 * f_prime2;
	 
						o.vertex.xy = h2.xy * o.vertex.w;
					}
					o.uv = TRANSFORM_TEX(v.uv, _MainTex);
					UNITY_TRANSFER_FOG(o,o.vertex);
					return o;
				}
				else {
					o.uv = TRANSFORM_TEX(v.uv, _MainTex);
					UNITY_TRANSFER_FOG(o,o.vertex);
					return o;
				}
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float2 h = i.uv.xy - float2(0.5, 0.5);
				float r = sqrt(h.x * h.x + h.y * h.y);
				//float f = r + _c1 * pow(r, 3) + _c2 * pow(r, 5);
				//float f_prime = _c1 * pow(f, 2) + _c2 * pow(f, 4) + pow(_c1, 2) * pow(f, 4) + pow(_c2, 2) * pow(f, 8) + 2 * _c1 * _c2 * pow(f, 6);
				//f_prime = f_prime  / (1 + 4 * _c1 * pow(f, 2) + 6 * _c2 * pow(f, 4));
				
				if(_lcaCorrection == 1 && _pincushionCorrection == 0) {
					float col_r = tex2D(_MainTex, float2(i.uv.x - 0.01, i.uv.y - 0.01)).r;
					float col_g = tex2D(_MainTex, i.uv).g;
					float col_b = tex2D(_MainTex, float2(i.uv.x + 0.01, i.uv.y + 0.01)).b;
					UNITY_APPLY_FOG(i.fogCoord, col);
					return fixed4(col_r, col_g, col_b, 1);
				}
				else if(_lcaCorrection == 0 && _pincushionCorrection == 1) {
					float f_prime = distort(_c1, _c2, r);
					float2 final = h + h * f_prime + 0.5;
					float4 col = tex2D(_MainTex, final);
					UNITY_APPLY_FOG(i.fogCoord, col);
					return col;
				}
				else if(_lcaCorrection == 1 && _pincushionCorrection == 1){
					float f_prime_r = distort(_c1 + 0.25, _c2, r);
					float f_prime_g = distort(_c1, _c2, r);
					float f_prime_b = distort(_c1 - 0.25, _c2, r);
					float2 final_r = h + h * f_prime_r + 0.5;
					float2 final_g = h + h * f_prime_g + 0.5;
					float2 final_b = h + h * f_prime_b + 0.5;
					float col_r = tex2D(_MainTex, final_r).r;
					float col_g = tex2D(_MainTex, final_g).g;
					float col_b = tex2D(_MainTex, final_b).b;
					UNITY_APPLY_FOG(i.fogCoord, col);
					return fixed4(col_r, col_g, col_b, 1);
				}
				else {
					UNITY_APPLY_FOG(i.fogCoord, col);
					return tex2D(_MainTex, i.uv);
				}
            }
            ENDCG
        }
    }
}

