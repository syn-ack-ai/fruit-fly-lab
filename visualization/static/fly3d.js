/*
 * 3D view of the digital fly's body.
 *
 * PROVENANCE
 *   D. OUR ENGINEERING, display only. Every pose comes from the body telemetry
 *   the server streams (fly/body/fly_body.py: position, heading, speed, turn
 *   rate, wing angle, leg extension, proboscis extension, airborne), which is
 *   itself driven only by descending-neuron and proboscis-motor-neuron output.
 *   Leg gait and wing-beat motion are procedural animation of those values;
 *   the connectome has no leg or wing motor neurons (they are in the VNC).
 *   Stimuli are drawn only to show where they are; the body never sees them.
 *
 * Anatomy: an adult female D. melanogaster, ~2.5 mm long, in millimetres.
 * World axes: three.js y is up; simulation (x, y) maps to three (x, -z), so
 * heading (counter-clockwise from +x, seen from above) is a rotation about +y.
 * In the fly's own frame +x is forward, +y up and +z its right side.
 */
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/three/OrbitControls.js';

const el = document.getElementById('fly3d');
const FOOD = new Set(['odor_vinegar', 'touch_leg_taste', 'taste_sugar']);
const DEG = Math.PI / 180;
const UP = new THREE.Vector3(0, 1, 0);

// ────────────────────────────── renderer / scene ──────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
el.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const BG = new THREE.Color('#10151d');
scene.background = BG;
scene.fog = new THREE.Fog(BG, 70, 260);

const camera = new THREE.PerspectiveCamera(36, 1, 0.05, 1500);
camera.position.set(-3.6, 2.3, 3.9);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 1.8;
controls.maxDistance = 220;
controls.maxPolarAngle = 89 * DEG;
controls.target.set(0, 0.55, 0);

scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x2b2218, 1.25));
const sun = new THREE.DirectionalLight(0xfff1e0, 2.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -7, right: 7, top: 7, bottom: -7, near: 1, far: 50 });
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.015;
scene.add(sun, sun.target);
const rim = new THREE.DirectionalLight(0x9dbbff, 0.9);
rim.position.set(-8, 5, -9);
scene.add(rim);

function canvasTex(w, h, draw, srgb = true) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

// ground: a lab arena floor, 1 mm grid with a bolder line every 10 mm
const groundTex = canvasTex(512, 512, (g, w) => {
  g.fillStyle = '#1a2029'; g.fillRect(0, 0, w, w);
  for (let i = 0; i < 5000; i++) {
    g.fillStyle = `rgba(255,255,255,${Math.random() * 0.035})`;
    g.fillRect(Math.random() * w, Math.random() * w, 2, 2);
  }
  g.strokeStyle = 'rgba(130,160,200,0.13)'; g.lineWidth = 2;
  for (let i = 1; i < 10; i++) {
    const p = i * w / 10;
    g.beginPath(); g.moveTo(p, 0); g.lineTo(p, w); g.stroke();
    g.beginPath(); g.moveTo(0, p); g.lineTo(w, p); g.stroke();
  }
  g.strokeStyle = 'rgba(150,185,230,0.32)'; g.lineWidth = 4; g.strokeRect(0, 0, w, w);
});
groundTex.wrapS = groundTex.wrapT = THREE.RepeatWrapping;
groundTex.repeat.set(80, 80);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(800, 800),
  new THREE.MeshStandardMaterial({ map: groundTex, roughness: 0.95, metalness: 0 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// ────────────────────────────────── the fly ──────────────────────────────────
// Colours and proportions follow photographs of an adult female
// D. melanogaster: glossy red-brown thorax covered in dark bristles, orange-tan
// head with large bright-red compound eyes, pale abdomen with dark dorsal
// tergite bands, long glassy wings folded flat past the abdomen, pale legs.
const mottled = (base, spots, n, a) => canvasTex(256, 256, (g, w) => {
  g.fillStyle = base; g.fillRect(0, 0, w, w);
  for (let i = 0; i < n; i++) {
    const x = Math.random() * w, y = Math.random() * w, r = 4 + Math.random() * 16;
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    gr.addColorStop(0, spots.replace('A', (a * Math.random()).toFixed(3)));
    gr.addColorStop(1, spots.replace('A', '0'));
    g.fillStyle = gr; g.fillRect(x - r, y - r, 2 * r, 2 * r);
  }
});
const mat = {
  thorax: new THREE.MeshPhysicalMaterial({ map: mottled('#9b5226', 'rgba(70,28,10,A)', 260, 0.55),
    roughness: 0.42, clearcoat: 0.45, clearcoatRoughness: 0.35 }),
  cuticle: new THREE.MeshPhysicalMaterial({ map: mottled('#b8692f', 'rgba(110,50,16,A)', 120, 0.4),
    roughness: 0.45, clearcoat: 0.3, clearcoatRoughness: 0.4 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x7a4420, roughness: 0.55 }),
  leg: new THREE.MeshStandardMaterial({ color: 0xc98a50, roughness: 0.55 }),
  tarsus: new THREE.MeshStandardMaterial({ color: 0xb07542, roughness: 0.55 }),
  bristle: new THREE.MeshStandardMaterial({ color: 0x160c06, roughness: 0.6 }),
  mouth: new THREE.MeshStandardMaterial({ color: 0xe8cfa6, roughness: 0.5 }),
};
// compound eye: bright red, facets only faintly visible at this scale
const eyeTex = canvasTex(256, 256, (g, w) => {
  g.fillStyle = '#b5121a'; g.fillRect(0, 0, w, w);
  const s = 7;
  for (let r = 0; r * s * 0.87 < w + s; r++) {
    for (let c = 0; c * s < w + s; c++) {
      const x = c * s + (r % 2) * s / 2, y = r * s * 0.87;
      const gr = g.createRadialGradient(x - 0.8, y - 0.8, 0, x, y, s * 0.55);
      gr.addColorStop(0, '#f23a33'); gr.addColorStop(0.75, '#d01b1f'); gr.addColorStop(1, '#8e0c12');
      g.fillStyle = gr; g.beginPath(); g.arc(x, y, s * 0.52, 0, 7); g.fill();
    }
  }
});
mat.eye = new THREE.MeshPhysicalMaterial({ map: eyeTex, roughness: 0.45, clearcoat: 0.35,
                                           clearcoatRoughness: 0.3, sheen: 0.4,
                                           sheenColor: new THREE.Color(0xff6a5a) });
// abdomen: cream, with dark bands on the dorsal side of each tergite only.
// Texture u runs around the body axis (u = 0 is dorsal), v along it.
const abdTex = canvasTex(256, 256, (g, w, h) => {
  g.fillStyle = '#d4b684'; g.fillRect(0, 0, w, h);
  const seg = 6;
  for (let x = 0; x < w; x++) {
    const dorsal = Math.pow(Math.max(0, Math.cos(2 * Math.PI * x / w)), 0.55);
    for (let i = 0; i < seg; i++) {
      const y0 = h * (0.12 + 0.8 * i / seg), y1 = h * (0.12 + 0.8 * (i + 1) / seg);
      const last = i >= seg - 2;
      const band = (y1 - y0) * (last ? 0.75 : 0.42) * (0.55 + 0.45 * dorsal);
      g.fillStyle = `rgba(52,31,15,${(last ? 0.92 : 0.85) * dorsal})`;
      g.fillRect(x, y1 - band, 1, band);
    }
  }
  const tip = g.createLinearGradient(0, h * 0.88, 0, h);
  tip.addColorStop(0, 'rgba(52,31,15,0)'); tip.addColorStop(1, 'rgba(52,31,15,0.8)');
  g.fillStyle = tip; g.fillRect(0, h * 0.88, w, h * 0.12);
});
mat.abdomen = new THREE.MeshStandardMaterial({ map: abdTex, roughness: 0.62 });

function ellipsoid(rx, ry, rz, m, seg = 32) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, seg, Math.round(seg * 0.75)), m);
  mesh.scale.set(rx, ry, rz);
  mesh.castShadow = true;
  return mesh;
}

// unit segment along +y from 0 to 1, placed between two points each frame
const segGeo = new THREE.CylinderGeometry(0.75, 1, 1, 8, 1).translate(0, 0.5, 0);
function segment(radius, m) {
  const s = new THREE.Mesh(segGeo, m);
  s.castShadow = true;
  s.userData.r = radius;
  return s;
}
const _d = new THREE.Vector3();
function place(seg, a, b, taper = 1) {
  _d.subVectors(b, a);
  const len = _d.length();
  seg.position.copy(a);
  seg.quaternion.setFromUnitVectors(UP, _d.normalize());
  seg.scale.set(seg.userData.r * taper, Math.max(len, 1e-4), seg.userData.r * taper);
}

const fly = new THREE.Group();          // world position + heading
const bodyRoot = new THREE.Group();     // height above ground, pitch, roll
fly.add(bodyRoot);
scene.add(fly);

// bristles (chaetae) on the surface of an ellipsoid part, swept backwards
function bristles(part, n, len, upperOnly, sweep = new THREE.Vector3(-1, 0.25, 0)) {
  const c = part.position, sc = part.scale;
  for (let i = 0; i < n; i++) {
    const u = Math.random() * 2 * Math.PI, v = Math.acos(1 - Math.random() * (upperOnly ? 1.1 : 2));
    const nrm = new THREE.Vector3(Math.sin(v) * Math.cos(u), Math.cos(v), Math.sin(v) * Math.sin(u));
    const base = new THREE.Vector3(nrm.x * sc.x, nrm.y * sc.y, nrm.z * sc.z).applyEuler(part.rotation).add(c);
    const dir = nrm.clone().multiplyScalar(0.45).add(sweep).normalize();
    const b = segment(len > 0.2 ? 0.006 : 0.0035, mat.bristle);
    place(b, base, base.clone().addScaledVector(dir, len * (0.7 + 0.6 * Math.random())));
    part.parent.add(b);
  }
}

// thorax (a tall hump, the highest point of the body) and scutellum
const thorax = ellipsoid(0.58, 0.5, 0.46, mat.thorax);
thorax.position.set(0.1, 0.07, 0);
bodyRoot.add(thorax);
const scutellum = ellipsoid(0.2, 0.1, 0.2, mat.thorax);
scutellum.position.set(-0.34, 0.36, 0);
bodyRoot.add(scutellum);
bristles(thorax, 70, 0.09, true);                            // microchaetae
for (const [x, z] of [[0.28, 0.13], [0.05, 0.14], [-0.15, 0.15], [0.4, 0.3], [-0.05, 0.36]]) {
  for (const side of [-1, 1]) {                             // macrochaetae
    const base = new THREE.Vector3(x, 0.07 + 0.5 * Math.sqrt(Math.max(0.05, 1 - (x - 0.1) ** 2 / 0.34 - z * z / 0.21)), side * z);
    const b = segment(0.0065, mat.bristle);
    place(b, base, base.clone().add(new THREE.Vector3(-0.28, 0.16, side * 0.05)));
    bodyRoot.add(b);
  }
}
for (const side of [-1, 1]) {                               // scutellar bristles
  for (const z of [0.05, 0.14]) {
    const base = new THREE.Vector3(-0.5, 0.42, side * z);
    const b = segment(0.0065, mat.bristle);
    place(b, base, base.clone().add(new THREE.Vector3(-0.34, 0.12, side * 0.08)));
    bodyRoot.add(b);
  }
}

// head: low at the front of the thorax, tilted down; big red eyes on the sides
const head = new THREE.Group();
head.position.set(0.68, -0.04, 0);
head.rotation.z = -0.28;
bodyRoot.add(head);
const capsule = ellipsoid(0.22, 0.29, 0.28, mat.cuticle);
head.add(capsule);
for (const side of [-1, 1]) {
  const eye = ellipsoid(0.2, 0.27, 0.16, mat.eye, 40);
  eye.position.set(0.02, 0.01, side * 0.17);
  eye.rotation.y = side * 0.18;
  head.add(eye);
  // antenna: orange funiculus with a feathery dark arista pointing forward
  const ant = new THREE.Group();
  ant.position.set(0.19, 0.06, side * 0.07);
  const a2 = ellipsoid(0.04, 0.04, 0.04, mat.cuticle, 12);
  ant.add(a2);
  const a3 = ellipsoid(0.07, 0.1, 0.05, mat.cuticle, 14);
  a3.position.set(0.05, -0.07, 0);
  ant.add(a3);
  const arista = segment(0.007, mat.bristle);
  const ar0 = new THREE.Vector3(0.08, -0.02, side * 0.02), ar1 = new THREE.Vector3(0.36, 0.14, side * 0.12);
  place(arista, ar0, ar1);
  ant.add(arista);
  for (let j = 1; j < 7; j++) {                             // arista branches
    const t = j / 7, p = ar0.clone().lerp(ar1, t);
    for (const up of [1, -1]) {
      const br = segment(0.0035, mat.bristle);
      place(br, p, p.clone().add(new THREE.Vector3(0.03, up * 0.05, side * 0.01)));
      ant.add(br);
    }
  }
  head.add(ant);
}
for (const [x, z] of [[-0.03, 0], [-0.08, 0.04], [-0.08, -0.04]]) {        // ocelli
  const o = ellipsoid(0.02, 0.013, 0.02, new THREE.MeshStandardMaterial({ color: 0xffb070, roughness: 0.3 }), 10);
  o.position.set(x, 0.29, z);
  head.add(o);
}
for (const side of [-1, 1]) {                               // vertical and orbital bristles
  for (const [x, y, z, dx, dy] of [[-0.1, 0.27, 0.1, -0.2, 0.2], [0.02, 0.28, 0.13, 0.08, 0.26],
                                   [0.1, 0.24, 0.16, 0.2, 0.18], [-0.14, 0.2, 0.2, -0.24, 0.12]]) {
    const b = segment(0.005, mat.bristle), p = new THREE.Vector3(x, y, side * z);
    place(b, p, p.clone().add(new THREE.Vector3(dx, dy, side * 0.06)));
    head.add(b);
  }
}

// proboscis: pale, rotates down from under the head and lengthens as it extends
const proboscis = new THREE.Group();
proboscis.position.set(0.1, -0.24, 0);
head.add(proboscis);
const probStem = segment(0.05, mat.mouth);
proboscis.add(probStem);
const labellum = new THREE.Group();
for (const side of [-1, 1]) {
  const lobe = ellipsoid(0.08, 0.04, 0.055, mat.mouth, 14);
  lobe.position.z = side * 0.045;
  labellum.add(lobe);
}
proboscis.add(labellum);

// abdomen: pale, sloping down towards the tail
const abdGeo = new THREE.SphereGeometry(1, 40, 28).rotateZ(-Math.PI / 2);
{ // taper towards the tip (-x)
  const p = abdGeo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), k = x < 0 ? 1 + 0.28 * x : 1;
    p.setY(i, p.getY(i) * k); p.setZ(i, p.getZ(i) * k);
  }
  abdGeo.computeVertexNormals();
}
const abdomen = new THREE.Mesh(abdGeo, mat.abdomen);
abdomen.scale.set(0.66, 0.34, 0.36);
abdomen.position.set(-0.66, -0.08, 0);
abdomen.rotation.z = 0.2;
abdomen.castShadow = true;
bodyRoot.add(abdomen);
bristles(abdomen, 50, 0.05, true, new THREE.Vector3(-1, 0.1, 0));

// wings: long, glassy grey with iridescence, folded flat past the abdomen
const WING_SCALE = 1.2;
function wingOutline() {
  const s = new THREE.Shape();
  s.moveTo(0, 0.05);
  s.bezierCurveTo(-0.6, 0.2, -1.6, 0.3, -2.15, 0.1);
  s.bezierCurveTo(-2.36, -0.05, -2.2, -0.42, -1.6, -0.48);
  s.bezierCurveTo(-1.0, -0.53, -0.45, -0.35, -0.12, -0.12);
  s.lineTo(0, 0.05);
  return s;
}
const WX0 = -2.4, WXS = 2.45, WY0 = -0.56, WYS = 0.9;   // wing coords -> texture
const wingTex = canvasTex(512, 192, (g, w, h) => {
  const P = (x, y) => [(x - WX0) / WXS * w, (1 - (y - WY0) / WYS) * h];
  g.fillStyle = 'rgba(150,158,168,0.2)'; g.fillRect(0, 0, w, h);
  g.strokeStyle = 'rgba(38,28,18,0.95)'; g.lineCap = 'round';
  const vein = (pts, lw) => {
    g.lineWidth = lw; g.beginPath(); g.moveTo(...P(...pts[0]));
    if (pts.length === 3) g.quadraticCurveTo(...P(...pts[1]), ...P(...pts[2]));
    else g.lineTo(...P(...pts[1]));
    g.stroke();
  };
  vein([[0, 0.05], [-1.2, 0.3], [-2.15, 0.1]], 4);            // costa
  vein([[-0.25, 0.03], [-1.1, 0.2], [-1.95, 0.17]], 2.2);      // L2
  vein([[-0.25, -0.02], [-1.2, 0.04], [-2.25, -0.02]], 2.2);   // L3
  vein([[-0.35, -0.08], [-1.2, -0.16], [-2.05, -0.3]], 2.2);   // L4
  vein([[-0.4, -0.14], [-1.0, -0.3], [-1.55, -0.47]], 2.2);    // L5
  vein([[-0.95, 0.03], [-0.95, -0.14]], 1.8);                  // anterior crossvein
  vein([[-1.38, -0.2], [-1.3, -0.36]], 1.8);                   // posterior crossvein
}, false);
wingTex.repeat.set(1 / WXS, 1 / WYS);
wingTex.offset.set(-WX0 / WXS, -WY0 / WYS);
const wingMat = new THREE.MeshPhysicalMaterial({
  map: wingTex, transparent: true, side: THREE.DoubleSide, depthWrite: false,
  roughness: 0.18, metalness: 0, iridescence: 0.75, iridescenceIOR: 1.33,
  iridescenceThicknessRange: [220, 620], specularIntensity: 0.45,
});
const wingGeo = new THREE.ShapeGeometry(wingOutline(), 24).rotateX(-Math.PI / 2)
  .scale(WING_SCALE, 1, WING_SCALE);
const blurMat = new THREE.MeshBasicMaterial({ color: 0xcfe0ff, transparent: true, opacity: 0.05,
                                              side: THREE.DoubleSide, depthWrite: false });
const wings = {};
for (const side of [-1, 1]) {                 // -1 left (-z), +1 right (+z)
  const yaw = new THREE.Group(), elev = new THREE.Group();
  yaw.position.set(0.24, 0.44 + (side > 0 ? 0.014 : 0), side * 0.19);
  yaw.add(elev);
  const m = new THREE.Mesh(wingGeo, wingMat);
  m.scale.z = -side;                         // leading edge lateral when folded, forward when spread
  m.renderOrder = 2;
  elev.add(m);
  // stroke-plane blur shown in flight (a real wing beats ~200 Hz). The
  // stroke sweeps yaw 28..152 deg from straight back; in the circle's own
  // angle that is 208..332 deg on the right side and 28..152 deg on the left.
  const fan = new THREE.Mesh(
    new THREE.CircleGeometry(2.25 * WING_SCALE, 36, (side > 0 ? 208 : 28) * DEG, 124 * DEG), blurMat);
  fan.rotation.x = -Math.PI / 2;
  fan.position.copy(yaw.position).add(new THREE.Vector3(0, 0.06, 0));
  fan.visible = false;
  fan.renderOrder = 1;
  bodyRoot.add(fan);
  bodyRoot.add(yaw);
  wings[side] = { yaw, elev, fan };
}

// legs: coxa, femur, tibia, tarsus; placed by 2-link IK so feet can plant
const LEGS = [];
const legSpec = [
  // name, hip (x, y, |z|), rest foot (x, |z|), femur, tibia, tarsus, tripod group
  ['T1', [0.44, -0.22, 0.12], [1.05, 0.8], 0.62, 0.56, 0.46, 0],
  ['T2', [0.2, -0.28, 0.15], [0.12, 1.3], 0.72, 0.7, 0.52, 1],
  ['T3', [-0.02, -0.26, 0.14], [-1.05, 1.1], 0.78, 0.76, 0.5, 0],
];
for (const side of [-1, 1]) {
  for (const [name, hip, foot, lf, lt, lta, grp] of legSpec) {
    const leg = {
      name: (side < 0 ? 'L' : 'R') + name, side,
      hip: new THREE.Vector3(hip[0], hip[1], side * hip[2]),
      rest: new THREE.Vector3(foot[0], 0, side * foot[1]),
      lf, lt, lta,
      group: (grp + (side < 0 ? 0 : 1)) % 2,   // tripods: L1 R2 L3 | R1 L2 R3
      coxa: segment(0.07, mat.leg), femur: segment(0.05, mat.leg),
      tibia: segment(0.034, mat.leg), tarsus: segment(0.022, mat.tarsus),
      knee: ellipsoid(0.042, 0.042, 0.042, mat.leg, 10),
      foot: new THREE.Vector3(),
    };
    for (const k of ['coxa', 'femur', 'tibia', 'tarsus', 'knee']) bodyRoot.add(leg[k]);
    LEGS.push(leg);
  }
}
const _T = new THREE.Vector3(), _A = new THREE.Vector3(), _K = new THREE.Vector3();
const _h = new THREE.Vector3(), _out = new THREE.Vector3();
function solveLeg(leg, foot) {
  // coxa hangs down and out from the hip; the trochanter is where the femur starts
  _T.copy(leg.hip).add(new THREE.Vector3(0.02, -0.13, leg.side * 0.06));
  place(leg.coxa, leg.hip, _T);
  // tarsus lies along the ground pointing outward from the ankle
  _out.set(foot.x - _T.x, 0, foot.z - _T.z);
  if (_out.lengthSq() < 1e-6) _out.set(0, 0, leg.side);
  _out.normalize();
  _A.copy(foot).addScaledVector(_out, -leg.lta * 0.8);
  _A.y += leg.lta * 0.45;
  // 2-link IK (femur, tibia) in the vertical plane through trochanter and ankle
  _h.set(_A.x - _T.x, 0, _A.z - _T.z);
  const r = _h.length();
  _h.divideScalar(r || 1);
  const dy = _A.y - _T.y, L1 = leg.lf, L2 = leg.lt;
  const D = Math.min(Math.hypot(r, dy), L1 + L2 - 1e-3);
  const alpha = Math.atan2(dy, r);
  const beta = Math.acos(Math.min(1, Math.max(-1, (L1 * L1 + D * D - L2 * L2) / (2 * L1 * D))));
  const th = alpha + beta;                                      // knee up
  _K.copy(_T).addScaledVector(_h, L1 * Math.cos(th)).addScaledVector(UP, L1 * Math.sin(th));
  const reach = new THREE.Vector3().copy(_T)
    .addScaledVector(_h, D * Math.cos(alpha)).addScaledVector(UP, D * Math.sin(alpha));
  place(leg.femur, _T, _K);
  place(leg.tibia, _K, reach, 0.95);
  place(leg.tarsus, reach, reach.clone().add(new THREE.Vector3().subVectors(foot, _A)), 1);
  leg.knee.position.copy(_K);
}

// ─────────────────────────────── stimuli (display) ───────────────────────────
const rockGeo = new THREE.IcosahedronGeometry(1, 4);
{
  const p = rockGeo.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const n = 1 + 0.08 * Math.sin(v.x * 5.1) * Math.cos(v.y * 4.3) + 0.05 * Math.sin(v.z * 7.7);
    p.setXYZ(i, v.x * n, v.y * n, v.z * n);
  }
  rockGeo.computeVertexNormals();
}
const rock = new THREE.Mesh(rockGeo, new THREE.MeshStandardMaterial({ color: 0x55555c, roughness: 0.92 }));
rock.castShadow = true;
rock.visible = false;
scene.add(rock);

const food = new THREE.Mesh(new THREE.SphereGeometry(0.6, 32, 20),
  new THREE.MeshPhysicalMaterial({ color: 0xffb13b, roughness: 0.08, transmission: 0.35,
                                   thickness: 0.5, clearcoat: 1 }));
food.scale.y = 0.45;
food.castShadow = true;
food.visible = false;
scene.add(food);

// what the camera sees, drawn in the fly's frame: its field of view and the
// looming region it is tracking
const camView = new THREE.Group();
camView.visible = false;
fly.add(camView);
{
  const R = 14, hf = 33 * DEG, vf = 20.5 * DEG;
  const c = (a, e) => new THREE.Vector3(R * Math.cos(e) * Math.cos(a), R * Math.sin(e), R * Math.cos(e) * Math.sin(a));
  const o = new THREE.Vector3(0.9, 0.7, 0);
  const pts = [c(-hf, vf), c(hf, vf), c(hf, -vf), c(-hf, -vf)].map(p => p.add(o));
  const lines = [o, pts[0], o, pts[1], o, pts[2], o, pts[3],
                 pts[0], pts[1], pts[1], pts[2], pts[2], pts[3], pts[3], pts[0]];
  camView.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(lines),
    new THREE.LineBasicMaterial({ color: 0x8fb3ff, transparent: true, opacity: 0.35 })));
}
const camBlob = new THREE.Mesh(new THREE.CircleGeometry(1, 40),
  new THREE.MeshBasicMaterial({ color: 0x8fb3ff, transparent: true, opacity: 0.35,
                                side: THREE.DoubleSide, depthWrite: false }));
camView.add(camBlob);

// ──────────────────────────── world (closed loop) ────────────────────────────
// Drawn from the world state the server streams (fly/world/world.py): arena,
// fruit, wind, the predator. The odour plume particles are a visual
// illustration of the wind-blown plume only; the fly's odour input is the
// server's plume model sampled at each antenna.
const worldG = new THREE.Group();
worldG.visible = false;
scene.add(worldG);
const grassTex = canvasTex(512, 512, (g, w) => {
  g.fillStyle = '#34402a'; g.fillRect(0, 0, w, w);
  for (let i = 0; i < 9000; i++) {
    const s = Math.random();
    g.fillStyle = s < 0.5 ? `rgba(90,120,60,${0.25 + Math.random() * 0.4})`
      : s < 0.8 ? `rgba(60,80,40,${0.3 + Math.random() * 0.4})` : `rgba(120,100,70,${0.2 + Math.random() * 0.3})`;
    const x = Math.random() * w, y = Math.random() * w;
    g.fillRect(x, y, 1 + Math.random() * 3, 1 + Math.random() * 6);
  }
});
grassTex.wrapS = grassTex.wrapT = THREE.RepeatWrapping;
const worldGround = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
  new THREE.MeshStandardMaterial({ map: grassTex, roughness: 1, metalness: 0 }));
worldGround.rotation.x = -Math.PI / 2;
worldGround.receiveShadow = true;
worldG.add(worldGround);
const walls = new THREE.Group();
worldG.add(walls);
let worldSize = 0;
function buildArena(size) {
  worldSize = size;
  const S = size + 600;
  worldGround.scale.set(S, S, 1);
  grassTex.repeat.set(S / 25, S / 25);
  walls.clear();
  const mat = new THREE.MeshStandardMaterial({ color: 0xa8b8c8, transparent: true, opacity: 0.16,
                                               side: THREE.DoubleSide, depthWrite: false });
  const edge = new THREE.LineBasicMaterial({ color: 0xcfe0f0, transparent: true, opacity: 0.5 });
  const H = 40;
  for (let k = 0; k < 4; k++) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(size, H), mat);
    const a = k * Math.PI / 2;
    wall.position.set(Math.cos(a) * size / 2, H / 2, -Math.sin(a) * size / 2);
    wall.rotation.y = a + Math.PI / 2;
    walls.add(wall);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(wall.geometry), edge);
    e.position.copy(wall.position); e.rotation.copy(wall.rotation);
    walls.add(e);
  }
}

const fruitMat = {
  sweet: new THREE.MeshPhysicalMaterial({ color: 0xc8261e, roughness: 0.35, clearcoat: 0.6 }),
  bitter: new THREE.MeshStandardMaterial({ color: 0x5e5a2a, roughness: 0.9 }),
  stem: new THREE.MeshStandardMaterial({ color: 0x4a3420, roughness: 0.8 }),
  leaf: new THREE.MeshStandardMaterial({ color: 0x3f7d2c, roughness: 0.7, side: THREE.DoubleSide }),
  spot: new THREE.MeshStandardMaterial({ color: 0x2b2a18, roughness: 1 }),
};
const fruitGeo = new THREE.SphereGeometry(1, 40, 28);
{   // apple-ish: dimpled top and bottom
  const p = fruitGeo.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const dimple = 1 - 0.18 * Math.exp(-(v.x * v.x + v.z * v.z) / 0.05);
    p.setXYZ(i, v.x * 1.05, v.y * 0.9 * dimple, v.z * 1.05);
  }
  fruitGeo.computeVertexNormals();
}
const fruits = [];
function makeFruit(kind) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(fruitGeo, fruitMat[kind]);
  body.castShadow = body.receiveShadow = true;
  g.add(body);
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.05, 0.35, 8), fruitMat.stem);
  stem.position.y = 0.9; stem.rotation.z = 0.2;
  g.add(stem);
  if (kind === 'sweet') {
    const leaf = new THREE.Mesh(new THREE.CircleGeometry(0.22, 12), fruitMat.leaf);
    leaf.scale.set(1.6, 0.7, 1); leaf.position.set(0.18, 1.0, 0); leaf.rotation.set(-1.1, 0.3, 0.4);
    g.add(leaf);
  } else {
    for (let i = 0; i < 9; i++) {                        // mould spots
      const s = new THREE.Mesh(new THREE.CircleGeometry(0.12 + Math.random() * 0.1, 10), fruitMat.spot);
      const th = Math.random() * Math.PI * 2, ph = 0.3 + Math.random() * 1.2;
      const n = new THREE.Vector3(Math.sin(ph) * Math.cos(th), Math.cos(ph) * 0.9, Math.sin(ph) * Math.sin(th));
      s.position.copy(n.clone().multiplyScalar(1.01)); s.lookAt(n.multiplyScalar(2));
      g.add(s);
    }
  }
  g.userData.kind = kind;
  worldG.add(g);
  return g;
}
function syncFruits(list) {
  while (fruits.length > list.length) worldG.remove(fruits.pop());
  list.forEach((f, i) => {
    if (!fruits[i] || fruits[i].userData.kind !== f.kind) {
      if (fruits[i]) worldG.remove(fruits[i]);
      fruits[i] = makeFruit(f.kind);
    }
    const r = f.r * (0.35 + 0.65 * Math.cbrt(Math.max(0, f.amount)));
    fruits[i].scale.setScalar(r);
    fruits[i].position.set(f.x, r * 0.9, -f.y);
    fruits[i].visible = f.amount > 0.001;
  });
}

// odour plume: particles leave each fruit and drift with the wind
const N_P = 2400;
const pPos = new Float32Array(N_P * 3), pAge = new Float32Array(N_P), pLife = new Float32Array(N_P);
const pSrc = new Int16Array(N_P);
const plumeGeo = new THREE.BufferGeometry();
plumeGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
const plume = new THREE.Points(plumeGeo, new THREE.PointsMaterial({
  color: 0xf6d47a, size: 2.2, sizeAttenuation: true, transparent: true, opacity: 0.22, depthWrite: false }));
plume.frustumCulled = false;
worldG.add(plume);
for (let i = 0; i < N_P; i++) { pAge[i] = 1e9; pLife[i] = 1; }
function stepPlume(dtSim, w) {
  const live = (w.fruits || []).map((f, i) => [f, i]).filter(([f]) => f.amount > 0.001);
  const U = w.wind.speed_mm_s, a = (w.wind.from_deg + 180) * DEG;
  const wx = U * Math.cos(a), wz = -U * Math.sin(a);
  for (let i = 0; i < N_P; i++) {
    pAge[i] += dtSim;
    const j = 3 * i;
    if (pAge[i] > pLife[i]) {
      if (!live.length) { pPos[j + 1] = -100; continue; }
      const [f] = live[(Math.random() * live.length) | 0];
      const th = Math.random() * Math.PI * 2;
      pPos[j] = f.x + Math.cos(th) * f.r; pPos[j + 1] = 1 + Math.random() * f.r * 1.4;
      pPos[j + 2] = -f.y - Math.sin(th) * f.r;
      pAge[i] = 0; pLife[i] = 1.5 + Math.random() * 4;
      continue;
    }
    // advection plus turbulent spread that grows with time since release
    const k = 25 + 60 * Math.min(1, pAge[i]);
    pPos[j] += (wx + (Math.random() - 0.5) * 2 * k) * dtSim;
    pPos[j + 2] += (wz + (Math.random() - 0.5) * 2 * k) * dtSim;
    pPos[j + 1] = Math.max(0.3, pPos[j + 1] + (Math.random() - 0.5) * 20 * dtSim);
    if (Math.abs(pPos[j]) > worldSize / 2 || Math.abs(pPos[j + 2]) > worldSize / 2) pAge[i] = 1e9;
  }
  plumeGeo.attributes.position.needsUpdate = true;
}

// predator: a dark bird-like shape that strikes at the fly
const predator = new THREE.Group();
{
  const m = new THREE.MeshStandardMaterial({ color: 0x1b1b22, roughness: 0.6 });
  const body = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), m);
  body.scale.set(1.5, 0.75, 0.8);
  predator.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.55, 20, 14), m);
  head.position.set(1.45, 0.25, 0);
  predator.add(head);
  const beak = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.7, 12), new THREE.MeshStandardMaterial({ color: 0xd8a326 }));
  beak.rotation.z = -Math.PI / 2; beak.position.set(2.1, 0.2, 0);
  predator.add(beak);
  for (const s of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.1, 10, 8), new THREE.MeshBasicMaterial({ color: 0xffe066 }));
    eye.position.set(1.75, 0.45, s * 0.33);
    predator.add(eye);
    const wing = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 2.6), m);
    wing.geometry.translate(0, 0, s * 1.3);
    wing.rotation.x = -Math.PI / 2;
    wing.userData.side = s;
    predator.add(wing);
  }
  predator.visible = false;
  predator.traverse(o => { if (o.isMesh) o.castShadow = true; });
  worldG.add(predator);
}
const predHist = [];        // [t_ms, predator|null]

// the fly's path
const TRAIL_N = 4000;
const trailPos = new Float32Array(TRAIL_N * 3);
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
trailGeo.setDrawRange(0, 0);
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0x9fd0ff, transparent: true, opacity: 0.45 }));
trail.frustumCulled = false;
worldG.add(trail);
let trailN = 0;
const trailLast = new THREE.Vector3(1e9, 0, 0);
function addTrail(p) {
  if (p.distanceTo(trailLast) < 1.5) return;
  trailLast.copy(p);
  if (trailN === TRAIL_N) { trailPos.copyWithin(0, 3); trailN--; }
  trailPos.set([p.x, p.y + 0.05, p.z], trailN * 3);
  trailN++;
  trailGeo.setDrawRange(0, trailN);
  trailGeo.attributes.position.needsUpdate = true;
}

let worldState = null;
function noteWorld(f) {
  const w = f.world;
  if (!w) {
    if (worldG.visible) {
      worldG.visible = false; ground.visible = true; worldState = null;
      scene.fog.near = 70; scene.fog.far = 260; controls.maxDistance = 220;
    }
    return;
  }
  if (!worldG.visible || w.size_mm !== worldSize) {
    buildArena(w.size_mm);
    worldG.visible = true; ground.visible = false;
    scene.fog.near = 500; scene.fog.far = 2200; controls.maxDistance = 1400;
    trailN = 0; trailGeo.setDrawRange(0, 0); predHist.length = 0;
  }
  worldState = w;
  syncFruits(w.fruits);
  predHist.push([f.t_ms, w.predator]);
  while (predHist.length > 400) predHist.shift();
}

function drawWorld(t, dt, pace, flyPos) {
  if (!worldState) return;
  stepPlume(dt * pace, worldState);
  addTrail(flyPos);
  let e = null;
  for (let i = predHist.length - 1; i >= 0; i--) if (predHist[i][0] <= t) { e = predHist[i]; break; }
  const p = e && e[1];
  predator.visible = !!p;
  if (p) {
    const dts = Math.max(0, (t - e[0]) / 1000);
    predator.position.set(p.x + p.vx * dts, Math.max(0, p.z + p.vz * dts), -(p.y + p.vy * dts));
    predator.scale.setScalar(p.r / 1.6);
    const v = new THREE.Vector3(p.vx, p.vz, -p.vy).normalize();
    predator.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), v);
    const flap = Math.sin(performance.now() / 1000 * 2 * Math.PI * 6) * 0.9;
    predator.children.forEach(c => { if (c.userData.side) c.rotation.x = -Math.PI / 2 + c.userData.side * flap; });
  }
  drawMinimap(flyPos);
}

// minimap: whole arena from above
const mini = document.createElement('canvas');
mini.className = 'fly3d-minimap';
mini.width = mini.height = 180;
mini.hidden = true;
el.appendChild(mini);
function drawMinimap(flyPos) {
  mini.hidden = !worldState;
  if (!worldState) return;
  const g = mini.getContext('2d'), W = mini.width, S = worldSize, k = (W - 16) / S;
  const X = (x) => 8 + (x + S / 2) * k, Y = (y) => 8 + (S / 2 - y) * k;
  g.clearRect(0, 0, W, W);
  g.fillStyle = 'rgba(30,40,26,0.85)'; g.fillRect(8, 8, S * k, S * k);
  g.strokeStyle = 'rgba(200,220,240,0.6)'; g.strokeRect(8, 8, S * k, S * k);
  g.strokeStyle = 'rgba(159,208,255,0.5)'; g.beginPath();
  for (let i = 0; i < trailN; i++) {
    const x = X(trailPos[3 * i]), y = Y(-trailPos[3 * i + 2]);
    if (i) g.lineTo(x, y); else g.moveTo(x, y);
  }
  g.stroke();
  for (const f of worldState.fruits) {
    if (f.amount <= 0.001) continue;
    g.fillStyle = f.kind === 'sweet' ? '#e0443a' : '#8a8440';
    g.beginPath(); g.arc(X(f.x), Y(f.y), Math.max(3, f.r * k), 0, 7); g.fill();
  }
  const p = worldState.predator;
  if (p) { g.fillStyle = '#ff3355'; g.beginPath(); g.arc(X(p.x), Y(p.y), 5, 0, 7); g.fill(); }
  g.fillStyle = '#ffffff'; g.beginPath(); g.arc(X(flyPos.x), Y(-flyPos.z), 3.5, 0, 7); g.fill();
  // wind arrow
  const wd = worldState.wind, a = (wd.from_deg + 180) * DEG;
  if (wd.speed_mm_s > 0) {
    const cx = W - 24, cy = 24, L = 12;
    g.strokeStyle = '#cfe0f0'; g.lineWidth = 2; g.beginPath();
    g.moveTo(cx - L * Math.cos(a), cy + L * Math.sin(a)); g.lineTo(cx + L * Math.cos(a), cy - L * Math.sin(a));
    g.stroke(); g.lineWidth = 1;
    g.fillStyle = '#cfe0f0'; g.font = '10px sans-serif'; g.fillText('wind', cx - 12, cy + 22);
  }
}

// ─────────────────────────── telemetry -> playback ───────────────────────────
// Each WebSocket frame carries the body pose for every simulated millisecond
// since the last frame. Playback runs a little behind the newest sample at
// the simulation's pace, so a 90 ms jump is shown as 90 ms, smoothly.
const track = [];          // [t, x, y, z, hd, spd, turn, wing, prob, leg, air, beh]
let latest = null;         // latest full frame (stimuli, pace)
let renderT = null;
let replay = null;         // {t, end, rate, label}
let expFrame = null;       // experiment-replay frame from the scrub bar
let lastAir = 0, takeoffT = null;
let autoSlowmo = true;
let forced = null;         // a fixed pose, for looking at the model
const stim = { rock: null, food: null };

window.addEventListener('fly-frame', (e) => {
  const f = e.detail;
  if (latest && f.t_ms < latest.t_ms - 1) {          // reset: start over
    track.length = 0; renderT = null; replay = null; stim.rock = stim.food = null;
  }
  latest = f;
  for (const s of f.track || []) {
    if (!track.length || s[0] > track[track.length - 1][0]) track.push(s);
  }
  const cut = track.length && track[track.length - 1][0] - 8000;
  let i = 0;
  while (i < track.length && track[i][0] < cut) i++;
  if (i) track.splice(0, i);
  noteStimuli(f);
  noteWorld(f);
});
window.addEventListener('fly-replay', (e) => { expFrame = e.detail; });

function sample(t) {
  if (!track.length) return null;
  if (t <= track[0][0]) return track[0];
  const n = track.length;
  if (t >= track[n - 1][0]) return track[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (track[m][0] <= t) lo = m; else hi = m; }
  const a = track[lo], b = track[hi], u = (t - a[0]) / (b[0] - a[0] || 1);
  const out = a.slice();
  for (let k = 1; k <= 9; k++) out[k] = a[k] + (b[k] - a[k]) * u;
  let dh = ((b[4] - a[4] + 540) % 360) - 180;          // shortest way round
  out[4] = a[4] + dh * u;
  out[0] = t;
  return out;
}

function flyPoseAt(t) {
  const s = sample(t);
  return s ? { x: s[1], y: s[2], hd: s[4] } : { x: 0, y: 0, hd: 0 };
}

// Stimuli are anchored in the world where the fly was when they appeared.
function noteStimuli(f) {
  const sts = f.stimuli || [];
  const loom = sts.find(s => s.distance_mm !== undefined);
  if (loom && loom.active && loom.collision_time_ms > f.t_ms) {
    const key = loom.collision_time_ms;
    if (!stim.rock || stim.rock.key !== key) {
      const speed = loom.distance_mm / ((loom.collision_time_ms - f.t_ms) / 1000);
      stim.rock = {
        key, speed, collision: loom.collision_time_ms, az: loom.azimuth_deg, el: loom.elevation_deg,
        size: loom.distance_mm * Math.tan(loom.half_angle_deg * DEG),
        from: f.t_ms - 30, anchor: flyPoseAt(f.t_ms),
      };
    }
  }
  const eating = sts.some(s => FOOD.has(s.modality) && s.active);
  if (eating && !stim.food) {
    const p = flyPoseAt(f.t_ms);
    stim.food = { x: p.x + 1.55 * Math.cos(p.hd * DEG), y: p.y + 1.55 * Math.sin(p.hd * DEG) };
  } else if (!eating && stim.food && !sts.some(s => FOOD.has(s.modality))) {
    stim.food = null;
  }
}

function worldDir(azDeg, elDeg, headingDeg) {
  const a = azDeg * DEG, e = elDeg * DEG;
  return new THREE.Vector3(Math.cos(e) * Math.cos(a), Math.sin(e), -Math.cos(e) * Math.sin(a))
    .applyAxisAngle(UP, headingDeg * DEG);
}

// ───────────────────────────────── animation ─────────────────────────────────
let gaitPhase = 0, gaitAmp = 0;
const stanceDir = new THREE.Vector3(1, 0, 0);
let prevFrameT = performance.now();
let follow = true;
const lastFly = new THREE.Vector3();
const hud = {};

function pose(s, wallT, dt) {
  // s: [t, x, y, z, hd, spd, turn, wing, prob, leg, air, beh]
  const [, x, y, z, hd, spd, turn, wing, prob, legExt, air, beh] = s;
  const airborne = air > 0.5;
  const freezing = /freez/.test(beh || '');
  fly.position.set(x, 0, -y);
  fly.rotation.y = hd * DEG;

  // --- gait: tripod, driven by forward speed and turning ---------------------
  const w = turn * DEG;                          // rad/s, + = counter-clockwise
  let moving = !airborne && !freezing && (Math.abs(spd) > 0.3 || Math.abs(turn) > 5);
  let speedAbs = 0;
  if (moving) {
    // how a planted foot moves in the body frame: -(v + w x r), at a mid radius
    const v = new THREE.Vector3(-spd, 0, 0).add(new THREE.Vector3(0, 0, 0.9).multiplyScalar(w));
    speedAbs = v.length();
    if (speedAbs > 1e-3) stanceDir.lerp(v.normalize(), Math.min(1, dt * 10)).normalize();
  }
  gaitAmp += ((moving ? 1 : 0) - gaitAmp) * Math.min(1, dt * 6);
  const stride = 0.32 + Math.min(0.3, Math.abs(spd) * 0.012);
  gaitPhase += (speedAbs * dt) / (2 * stride) * (moving ? 1 : 0);

  let standH = 0.6;
  if (freezing) standH = 0.46;
  standH += legExt * 0.5 * (airborne ? 0 : 1);
  bodyRoot.position.y = standH + (airborne ? z : 0) +
    0.018 * gaitAmp * Math.sin(gaitPhase * 4 * Math.PI);
  bodyRoot.rotation.z = airborne ? 0.18 : (-0.05 + 0.1 * legExt);   // nose up when jumping
  bodyRoot.rotation.x = -0.12 * gaitAmp * Math.max(-1, Math.min(1, turn / 300));

  const ground = -bodyRoot.position.y;
  for (const leg of LEGS) {
    const f = leg.foot.copy(leg.rest);
    if (airborne) {                                          // legs trail and dangle
      f.set(leg.rest.x * 0.55 - 0.15, -0.72, leg.rest.z * 0.55);
    } else {
      f.y = ground;
      const q = (gaitPhase + leg.group * 0.5) % 1;
      if (q < 0.5) {
        f.addScaledVector(stanceDir, stride * (q / 0.5 - 0.5) * gaitAmp);
      } else {
        const u = (q - 0.5) / 0.5;
        f.addScaledVector(stanceDir, stride * (0.5 - u) * gaitAmp);
        f.y += 0.26 * Math.sin(Math.PI * u) * gaitAmp;
      }
      // takeoff: middle legs push down and back, the others lift away
      if (legExt > 0.01) {
        if (leg.name.endsWith('T2')) f.x -= 0.25 * legExt;
        else f.y += 0.35 * legExt;
      }
    }
    solveLeg(leg, f);
  }

  // --- wings -------------------------------------------------------------------
  for (const side of [-1, 1]) {
    const W = wings[side];
    let yaw, elev;
    if (airborne) {
      const beat = Math.sin(wallT * 2 * Math.PI * 17);        // shown slowed; real ~200 Hz
      yaw = 90 + 62 * beat;
      elev = 8 + 10 * Math.cos(wallT * 2 * Math.PI * 17);
      W.fan.visible = true;
    } else {
      const raise = Math.min(1, wing / 90);
      yaw = -4 + 37 * raise;                  // folded: tips converge, wings overlap
      elev = 3 + 76 * raise;
      W.fan.visible = false;
    }
    W.yaw.rotation.y = side * yaw * DEG;
    W.elev.rotation.z = -elev * DEG;
  }

  // --- proboscis ------------------------------------------------------------------
  const pe = Math.max(0, Math.min(1, prob));
  proboscis.rotation.z = (-70 + 90 * pe) * DEG;              // folded back -> down/forward
  const len = 0.14 + 0.3 * pe;
  place(probStem, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -len, 0));
  labellum.position.set(0, -len - 0.02, 0);
  labellum.scale.setScalar(0.6 + 0.6 * pe);
}

function drawStimuli(t, s) {
  // looming rock, world-anchored where the fly was when it was thrown
  const R = stim.rock;
  if (R && t >= R.from && t < R.collision) {
    const d = R.speed * (R.collision - t) / 1000;
    const dir = worldDir(R.az, R.el, R.anchor.hd);
    rock.position.set(R.anchor.x, 0.7, -R.anchor.y).addScaledVector(dir, d + R.size);
    rock.position.y = Math.max(rock.position.y, R.size * 0.98);    // not through the floor
    rock.scale.setScalar(R.size);
    rock.visible = d > 0;
  } else {
    rock.visible = false;
  }
  if (stim.food) {
    food.position.set(stim.food.x, 0.27, -stim.food.y);
    food.visible = true;
  } else food.visible = false;

  // camera: its field of view in front of the fly, and the tracked region
  const cam = latest && (latest.stimuli || []).find(x => x.source === 'camera');
  camView.visible = !!cam && !replay && !expFrame && !forced;
  if (cam) {
    camBlob.visible = cam.active && cam.half_angle_deg > 0;
    if (camBlob.visible) {
      const Rv = 13.5, dir = worldDir(cam.azimuth_deg, cam.elevation_deg, 0);
      camBlob.position.set(0.9, 0.7, 0).addScaledVector(dir, Rv);
      camBlob.lookAt(fly.localToWorld(new THREE.Vector3(0.9, 0.7, 0)));
      camBlob.scale.setScalar(Rv * Math.tan(cam.half_angle_deg * DEG));
      camBlob.material.color.set(cam.expansion_rate_deg_s > 0 ? 0xff6b81 : 0x8fb3ff);
    }
  }
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - prevFrameT) / 1000);
  prevFrameT = now;
  if (document.hidden || !el.offsetParent) return;

  const pace = latest ? (latest.pace > 0 ? latest.pace : (latest.realtime_factor || 1)) : 1;
  let s;
  if (forced) { s = forced; renderT = 0; }
  else if (expFrame) {                              // experiment scrub bar
    const b = expFrame.body || {};
    s = [expFrame.t_ms, b.x_mm || 0, b.y_mm || 0, b.z_mm || 0, b.heading_deg || 0, 0, 0,
         b.wing_angle_deg || 0, b.proboscis_extension || 0, 0, b.airborne ? 1 : 0, b.behaviour];
    renderT = expFrame.t_ms;
  } else if (replay) {                              // slow-motion replay
    replay.t += dt * 1000 * replay.rate;
    if (replay.t >= replay.end) { replay = null; renderT = null; }
    else { renderT = replay.t; }
    s = sample(renderT ?? 0);
  }
  if (!forced && !expFrame && !replay) {
    if (track.length) {
      const newest = track[track.length - 1][0];
      const running = latest && latest.realtime_factor;
      renderT = renderT == null ? newest - 60 * pace : renderT + dt * 1000 * pace;
      if (!running || renderT > newest) renderT = newest;
      if (renderT < newest - 500 * Math.max(pace, 0.25)) renderT = newest - 60 * pace;
      s = sample(renderT);
      // auto slow-motion replay of each takeoff, once it has landed
      const air = s[10] > 0.5;
      if (air && !lastAir) takeoffT = renderT;
      if (!air && lastAir && takeoffT != null && autoSlowmo && pace >= 0.5) {
        startReplay(takeoffT - 260, renderT + 120, 0.1, 'takeoff');
      }
      lastAir = air;
    }
  }
  if (!s) s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'resting'];

  pose(s, now / 1000, dt);
  drawStimuli(renderT ?? 0, s);
  drawWorld(renderT ?? 0, dt, pace, fly.position);

  // follow the fly, keeping the viewer's chosen orbit offset
  if (follow) {
    const p = new THREE.Vector3(fly.position.x, 0.55 + (s[10] > 0.5 ? s[3] : 0), fly.position.z);
    // lock on (a flying fly covers ~4 mm per frame; smoothing would lose it)
    const delta = p.clone().sub(controls.target);
    controls.target.add(delta);
    camera.position.add(delta);
  }
  // the floor follows the fly in whole 10 mm tiles, so the grid never slides
  ground.position.set(Math.round(fly.position.x / 10) * 10, 0, Math.round(fly.position.z / 10) * 10);
  sun.target.position.copy(controls.target);
  sun.position.copy(controls.target).add(new THREE.Vector3(5, 12, 4));
  controls.update();
  renderer.render(scene, camera);

  hud.banner.hidden = !replay || replay.rate === 0;
  if (replay) hud.banner.textContent =
    `SLOW-MOTION REPLAY ×${replay.rate}  ·  t = ${replay.t.toFixed(0)} ms`;
  hud.clock.textContent = `${(renderT ?? 0).toFixed(0)} ms` +
    (s[11] ? `  ·  ${s[11]}` : '');
}

window.__fly3d = {                 // for scripted checks and screenshots
  replay: (t0, t1, rate = 0.1) => startReplay(t0, t1, rate, 'manual'),
  track: () => track.length ? [track[0][0], track[track.length - 1][0]] : null,
  events: () => { const e = []; let air = 0;
    for (const s of track) { if (s[10] !== air) e.push([s[0], s[10] ? 'takeoff' : 'landed']); air = s[10]; }
    return e; },
  rock: () => stim.rock,
  pose: (o) => {                    // pin a pose, e.g. {prob: 1} or {wing: 90}; null to release
    forced = o ? [0, 0, 0, o.z || 0, 0, o.spd || 0, o.turn || 0, o.wing || 0, o.prob || 0,
                  o.leg || 0, o.air ? 1 : 0, o.beh || 'resting'] : null;
    if (forced) { fly.position.set(0, 0, 0); stim.rock = stim.food = null; }
  },
  at: (t) => { replay = { t, end: Infinity, rate: 0, label: 'frozen' }; },   // hold a moment
  live: () => { replay = null; renderT = null; },
  view: (name) => {                                   // camera presets
    const o = { side: [0.2, 0.5, 5.2], front: [5.2, 0.9, 0.3], top: [0.01, 6, 0.01],
                oblique: [-3.6, 2.3, 3.9] }[name];
    follow = true;
    controls.target.set(fly.position.x, 0.55, fly.position.z);
    camera.position.set(fly.position.x + o[0], o[1], fly.position.z + o[2]);
  },
};

function startReplay(t0, t1, rate, label) {
  if (!track.length) return;
  const first = track[0][0], last = track[track.length - 1][0];
  replay = { t: Math.max(first, t0), end: Math.min(last, t1), rate, label };
  if (replay.end <= replay.t) replay = null;
}

// ────────────────────────────────── overlay ──────────────────────────────────
function overlay() {
  const bar = document.createElement('div');
  bar.className = 'fly3d-bar';
  bar.innerHTML = `
    <button data-a="replay" title="Replay the last second in slow motion">↺ slow-mo</button>
    <button data-a="auto" class="on" title="Replay every takeoff in slow motion">auto slow-mo</button>
    <button data-a="follow" class="on" title="Keep the camera on the fly">follow</button>
    <button data-a="home" title="Reset the view">⌂</button>
    <button data-a="overview" title="See the whole world from above">overview</button>`;
  bar.onclick = (e) => {
    const a = e.target.dataset.a;
    if (a === 'replay' && track.length) {
      const last = track[track.length - 1][0];
      startReplay(last - 1000, last, 0.1, 'manual');
    } else if (a === 'auto') {
      autoSlowmo = !autoSlowmo; e.target.classList.toggle('on', autoSlowmo);
    } else if (a === 'follow') {
      follow = !follow; e.target.classList.toggle('on', follow);
    } else if (a === 'overview') {
      follow = false; bar.querySelector('[data-a=follow]').classList.remove('on');
      const S = worldSize || 60;
      controls.target.set(0, 0, 0);
      camera.position.set(0, S * 0.95, S * 0.55);
    } else if (a === 'home') {
      follow = true; bar.querySelector('[data-a=follow]').classList.add('on');
      controls.target.set(fly.position.x, 0.55, fly.position.z);
      camera.position.set(fly.position.x - 3.6, 2.3, fly.position.z + 3.9);
    }
  };
  el.appendChild(bar);
  hud.banner = document.createElement('div');
  hud.banner.className = 'fly3d-banner';
  hud.banner.hidden = true;
  el.appendChild(hud.banner);
  hud.clock = document.createElement('div');
  hud.clock.className = 'fly3d-clock';
  el.appendChild(hud.clock);
  const note = document.createElement('div');
  note.className = 'fly3d-note';
  note.textContent = 'body driven only by descending-neuron output · drag to orbit · scroll to zoom';
  el.appendChild(note);
}

function resize() {
  const w = el.clientWidth, h = el.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(el);
overlay();
resize();
requestAnimationFrame(frame);
