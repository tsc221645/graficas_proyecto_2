// ===============================================================
// Ray Tracer Diorama voxel (Rust std + rayon + tilemap ASCII)
// Controlas la colocación de cubos editando un mapa de caracteres.
// W=Water, S=Sand, G=Grass, D=Dirt, M=Wood, L=Leaves
// Agua con reflexión/refracción (ior=1.33). 120 frames (video).
// ===============================================================

use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs::{create_dir_all, File};
use std::io::Write;

// ------------------------- Math -------------------------------
#[derive(Clone, Copy, Debug, Default)]
struct Vec3 { x: f32, y: f32, z: f32 }

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    fn dot(self, o: Self) -> f32 { self.x*o.x + self.y*o.y + self.z*o.z }
    fn cross(self, o: Self) -> Self {
        Self::new(self.y*o.z - self.z*o.y, self.z*o.x - self.x*o.z, self.x*o.y - self.y*o.x)
    }
    fn len(self) -> f32 { self.dot(self).sqrt() }
    fn norm(self) -> Self { let l = self.len().max(1e-8); self / l }
    fn reflect(self, n: Self) -> Self { self - n * (2.0 * self.dot(n)) }
    fn refract(self, n: Self, eta: f32) -> Option<Self> {
        let cosi = (-self.dot(n)).clamp(-1.0, 1.0);
        let sint2 = eta*eta * (1.0 - cosi*cosi);
        if sint2 > 1.0 { return None; }
        let cost = (1.0 - sint2).sqrt();
        Some(self*eta + n*(eta*cosi - cost))
    }
}
use std::ops::{Add, Sub, Mul, Div, Neg};
impl Add for Vec3 { type Output=Self; fn add(self,o:Self)->Self{Self::new(self.x+o.x,self.y+o.y,self.z+o.z)} }
impl Sub for Vec3 { type Output=Self; fn sub(self,o:Self)->Self{Self::new(self.x-o.x,self.y-o.y,self.z-o.z)} }
impl Mul<f32> for Vec3 { type Output=Self; fn mul(self,s:f32)->Self{Self::new(self.x*s,self.y*s,self.z*s)} }
impl Mul<Vec3> for Vec3 { type Output=Self; fn mul(self,o:Self)->Self{Self::new(self.x*o.x,self.y*o.y,self.z*o.z)} }
impl Div<f32> for Vec3 { type Output=Self; fn div(self,s:f32)->Self{Self::new(self.x/s,self.y/s,self.z/s)} }
impl Neg for Vec3 { type Output=Self; fn neg(self)->Self{Self::new(-self.x,-self.y,-self.z)} }

type Color = Vec3;
fn clamp01(x:f32)->f32 { x.max(0.0).min(1.0) }
fn to_u8(x:f32)->u8 { (clamp01(x).powf(1.0/2.2)*255.0 + 0.5) as u8 }

// ------------------------- Ray -------------------------------
#[derive(Clone, Copy, Debug)]
struct Ray { o: Vec3, d: Vec3 }

#[derive(Clone)]
enum Texture {
    Solid(Color),
    Image(Image),  // <- imagen cargada
}

impl Texture {
    fn sample(&self, uv:(f32,f32)) -> Color {
        match self {
            Texture::Solid(c) => *c,
            Texture::Image(img) => img.sample(uv.0, uv.1),
        }
    }
}

#[derive(Clone)]
struct Material {
    albedo: Color,
    specular: f32,
    reflectivity: f32,
    transparency: f32,
    ior: f32,
    tex: Texture,   // <- aquí usamos textura en vez de solo color
}

impl Material {
    fn base_color(&self, _uv:(f32,f32)) -> Color {
        self.tex.sample(_uv) * self.albedo
    }
    fn base_color_no_uv(&self) -> Color {
        self.tex.sample((0.0,0.0)) * self.albedo
    }
}

#[derive(Clone)]
struct Image {
    w: usize,
    h: usize,
    data: Vec<u8>,
}

impl Image {
    fn read_ppm(path: &str) -> std::io::Result<Self> {
        use std::io::{BufRead, Read};
        let mut f = std::fs::File::open(path)?;
        let mut buf = std::io::BufReader::new(&f);

        // Leer cabecera (P6)
        let mut magic = String::new();
        buf.read_line(&mut magic)?;
        assert!(magic.trim() == "P6", "PPM debe ser P6");

        // Leer width height
        let mut dims = String::new();
        buf.read_line(&mut dims)?;
        let parts: Vec<_> = dims.split_whitespace().collect();
        let w: usize = parts[0].parse().unwrap();
        let h: usize = parts[1].parse().unwrap();

        // Leer maxval
        let mut maxv = String::new();
        buf.read_line(&mut maxv)?;
        let maxv: usize = maxv.trim().parse().unwrap();
        assert!(maxv == 255);

        // Leer datos binarios (RGB)
        let mut data = Vec::new();
        buf.read_to_end(&mut data)?;

        Ok(Self { w, h, data })
    }
    fn sample(&self, u: f32, v: f32) -> Color {
        // wrap UV a [0,1]
        let mut uu = u.fract(); if uu < 0.0 { uu += 1.0; }
        let mut vv = v.fract(); if vv < 0.0 { vv += 1.0; }

        let x = (uu * (self.w as f32 - 1.0)) as usize;
        let y = ((1.0 - vv) * (self.h as f32 - 1.0)) as usize;
        let i = 3 * (y * self.w + x);

        Color::new(
            self.data[i] as f32 / 255.0,
            self.data[i+1] as f32 / 255.0,
            self.data[i+2] as f32 / 255.0,
        )
    }
}


// ------------------------- Geometría -------------------------
#[derive(Clone)]
struct Hit { t: f32, p: Vec3, n: Vec3, uv:(f32,f32), mat: Material }


trait Intersect { fn intersect(&self, ray:&Ray) -> Option<Hit>; }

#[derive(Clone)]
struct Aabb { min: Vec3, max: Vec3, mat: Material }
impl Aabb { fn new(min:Vec3, max:Vec3, mat:Material) -> Self { Self{min,max,mat} } }

impl Intersect for Aabb {
    fn intersect(&self, ray:&Ray) -> Option<Hit> {
        let inv = Vec3::new(1.0/ray.d.x, 1.0/ray.d.y, 1.0/ray.d.z);
        let (tx1, tx2) = ((self.min.x - ray.o.x)*inv.x, (self.max.x - ray.o.x)*inv.x);
        let (ty1, ty2) = ((self.min.y - ray.o.y)*inv.y, (self.max.y - ray.o.y)*inv.y);
        let (tz1, tz2) = ((self.min.z - ray.o.z)*inv.z, (self.max.z - ray.o.z)*inv.z);

        let (txmin, txmax) = (tx1.min(tx2), tx1.max(tx2));
        let (tymin, tymax) = (ty1.min(ty2), ty1.max(ty2));
        let (tzmin, tzmax) = (tz1.min(tz2), tz1.max(tz2));
        let tmin = txmin.max(tymin).max(tzmin);
        let tmax = txmax.min(tymax).min(tzmax);
        if tmax < tmin || tmax < 1e-4 { return None; }
        let t = if tmin > 1e-4 { tmin } else { tmax };

        let p = ray.o + ray.d * t;
        let n = if (t - txmin).abs() < 1e-5 {
            Vec3::new(if tx1 < tx2 {-1.0}else{1.0},0.0,0.0)
        } else if (t - tymin).abs() < 1e-5 {
            Vec3::new(0.0, if ty1 < ty2 {-1.0}else{1.0},0.0)
        } else {
            Vec3::new(0.0,0.0, if tz1 < tz2 {-1.0}else{1.0})
        };

        // UV por cara (proyección cúbica)
        let (u, v) = if n.x.abs() > 0.5 {
            // ±X → usa (z,y)
            let u = (p.z - self.min.z) / (self.max.z - self.min.z);
            let v = (p.y - self.min.y) / (self.max.y - self.min.y);
            (u, v)
        } else if n.y.abs() > 0.5 {
            // ±Y → usa (x,z)
            let u = (p.x - self.min.x) / (self.max.x - self.min.x);
            let v = (p.z - self.min.z) / (self.max.z - self.min.z);
            (u, v)
        } else {
            // ±Z → usa (x,y)
            let u = (p.x - self.min.x) / (self.max.x - self.min.x);
            let v = (p.y - self.min.y) / (self.max.y - self.min.y);
            (u, v)
        };

        Some(Hit { t, p, n, uv:(u,v), mat: self.mat.clone() })
    }
}

// -------------------------- Luz ------------------------------
struct PointLight { pos: Vec3, color: Color, intensity: f32 }

// -------------------------- Sky ------------------------------
enum Sky { Gradient { top: Color, bottom: Color } }
impl Sky {
    fn sample(&self, dir:Vec3) -> Color {
        match self {
            Sky::Gradient { top, bottom } => {
                let t = 0.5*(dir.y + 1.0);
                *bottom*(1.0 - t) + *top*t
            }
        }
    }
}

// ------------------------- Cámara ----------------------------
struct Camera { pos: Vec3, target: Vec3, up: Vec3, fov_deg: f32, aspect: f32 }
impl Camera {
    fn ray_for(&self, x:f32, y:f32, w:f32, h:f32) -> Ray {
        let fwd = (self.target - self.pos).norm();
        let right = fwd.cross(self.up).norm();
        let up = right.cross(fwd).norm();
        let fov = self.fov_deg.to_radians();
        let px = ((x + 0.5) / w) * 2.0 - 1.0;
        let py = 1.0 - ((y + 0.5) / h) * 2.0;
        let tan = (fov*0.5).tan();
        let dir = (fwd + right*(px* tan * self.aspect) + up*(py* tan)).norm();
        Ray { o: self.pos, d: dir }
    }
}

// ------------------------- Escena ----------------------------
enum Object { Box(Aabb) }
impl Intersect for Object { fn intersect(&self, ray:&Ray) -> Option<Hit> { match self { Object::Box(b) => b.intersect(ray) } } }

struct Scene { objects: Vec<Object>, lights: Vec<PointLight>, sky: Sky }
impl Scene {
    fn trace(&self, ray:&Ray, depth:i32) -> Color {
        // hit más cercano
        let mut best: Option<Hit> = None;
        let mut best_t = f32::INFINITY;
        for o in &self.objects {
            if let Some(h) = o.intersect(ray) { if h.t < best_t && h.t > 1e-4 { best_t = h.t; best = Some(h); } }
        }
        if best.is_none() { return self.sky.sample(ray.d); }
        let hit = best.unwrap();

        let view = (-ray.d).norm();
        let base = hit.mat.base_color(hit.uv);


        // luz directa + sombras
        let mut color = base * 0.08;
        for l in &self.lights {
            let ldir = l.pos - hit.p;
            let dist = ldir.len();
            let ldirn = ldir / dist;

            // sombra
            let shadow_ray = Ray { o: hit.p + hit.n*1e-3, d: ldirn };
            let mut occluded = false;
            for o in &self.objects {
                if let Some(h) = o.intersect(&shadow_ray) { if h.t > 1e-4 && h.t < dist - 1e-3 { occluded = true; break; } }
            }
            if !occluded {
                let ndotl = hit.n.dot(ldirn).max(0.0);
                let diff = base * (ndotl * l.intensity / (dist*dist).max(1.0));
                let hvec = (ldirn + view).norm();
                let spec = hit.mat.specular * hvec.dot(hit.n).max(0.0).powf(64.0);
                color = color + diff * l.color + Color::new(spec,spec,spec);
            }
        }

        // reflexión / refracción (agua)
        if depth <= 0 { return color; }
        let mut refl_col = Color::new(0.0,0.0,0.0);
        let mut refr_col = Color::new(0.0,0.0,0.0);
        let mut kr = hit.mat.reflectivity;

        if hit.mat.transparency > 0.0 {
            let mut cosi = (-ray.d).dot(hit.n).clamp(-1.0, 1.0);
            let mut etai = 1.0; let mut etat = hit.mat.ior;
            let n = if cosi > 0.0 { hit.n } else { cosi=-cosi; std::mem::swap(&mut etai,&mut etat); -hit.n };
            let eta = etai/etat;
            let r0 = ((etai-etat)/(etai+etat)).powi(2);
            let fresnel = r0 + (1.0-r0)*(1.0-cosi).powi(5);
            if let Some(tdir) = ray.d.refract(n, eta) {
                let refr_ray = Ray { o: hit.p - n*1e-3, d: tdir.norm() };
                refr_col = self.trace(&refr_ray, depth-1);
            } else { kr = 1.0; }
            kr = (kr+fresnel).min(1.0);
        }
        if hit.mat.reflectivity > 0.0 || kr > 0.0 {
            let rdir = ray.d.reflect(hit.n).norm();
            let rray = Ray { o: hit.p + hit.n*1e-3, d: rdir };
            refl_col = self.trace(&rray, depth-1);
        }
        let refl_w = kr.max(hit.mat.reflectivity).min(1.0);
        let refr_w = hit.mat.transparency.min(1.0 - refl_w);
        let base_w = (1.0 - refl_w - refr_w).max(0.0);
        color * base_w + refl_col * refl_w + refr_col * refr_w
    }
}

// ------------------------- I/O -------------------------------
fn write_frame(buffer:&[Color], w:usize, h:usize, path:&str) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    write!(f,"P6\n{} {}\n255\n",w,h)?;
    for c in buffer { f.write_all(&[to_u8(c.x),to_u8(c.y),to_u8(c.z)])?; }
    Ok(())
}

// --------------------- Helpers de colocación -----------------
fn add_cube(objs:&mut Vec<Object>, x:f32,y:f32,z:f32, sx:f32,sy:f32,sz:f32, mat:&Material){
    let min = Vec3::new(x,y,z);
    let max = Vec3::new(x+sx, y+sy, z+sz);
    objs.push(Object::Box(Aabb::new(min,max,mat.clone())));
}
// cubo 1x1x1 en grilla
fn put(objs:&mut Vec<Object>, gx:i32, gy:i32, gz:i32, mat:&Material){
    add_cube(objs, gx as f32, gy as f32, gz as f32, 1.0, 1.0, 1.0, mat);
}

// Coloca un tilemap ASCII en Y=layer_y (cada char => un bloque)
fn place_tilemap(
    objs:&mut Vec<Object>,
    origin_x:i32, origin_z:i32,  // esquina superior izquierda del mapa
    layer_y:f32,                 // altura del bloque (ej: -0.98 para arena, -0.90 agua)
    h:f32,                       // grosor (altura) del bloque
    map:&[&str],
    mat_g:&Material, mat_d:&Material, mat_s:&Material, mat_w:&Material, mat_m:&Material, mat_l:&Material
){
    for (rz, row) in map.iter().enumerate() {
        for (rx, ch) in row.chars().enumerate() {
            let gx = origin_x + rx as i32;
            let gz = origin_z + rz as i32;
            match ch {
                'G' => add_cube(objs, gx as f32, layer_y, gz as f32, 1.0, h, 1.0, mat_g),
                'D' => add_cube(objs, gx as f32, layer_y, gz as f32, 1.0, h, 1.0, mat_d),
                'S' => add_cube(objs, gx as f32, layer_y, gz as f32, 1.0, h, 1.0, mat_s),
                'W' => add_cube(objs, gx as f32, layer_y, gz as f32, 1.0, h, 1.0, mat_w),
                'M' => add_cube(objs, gx as f32, layer_y, gz as f32, 1.0, h, 1.0, mat_m),
                'L' => add_cube(objs, gx as f32, layer_y, gz as f32, 1.0, h, 1.0, mat_l),
                _   => {}
            }
        }
    }
}

// --------------------------- Main ----------------------------
fn main() -> std::io::Result<()> {
    let width=800usize; let height=600usize; let aspect=width as f32/height as f32;

    // --- Materiales (colores lisos) ---
    //let grass = Material{albedo:Color::new(1.,1.,1.),specular:0.2,reflectivity:0.05,transparency:0.,ior:1.,color:Color::new(0.12,0.55,0.12)};
    let grass_img = Image::read_ppm("textures/grass.ppm")?;
    let sand_img = Image::read_ppm("textures/sand.ppm")?;
    let wood_img = Image::read_ppm("textures/wood.ppm")?;
    let leaves_img = Image::read_ppm("textures/leaves.ppm")?;
    let grass = Material {
        albedo: Color::new(1.0,1.0,1.0),
        specular: 0.2,
        reflectivity: 0.00,
        transparency: 0.0,
        ior: 1.0,
        tex: Texture::Image(grass_img.clone()),
    };
    let sand = Material {
        albedo: Color::new(1.0,1.0,1.0),
        specular: 0.2,
        reflectivity: 0.00,
        transparency: 0.0,
        ior: 1.0,
        tex: Texture::Image(sand_img),
    };
    let water = Material {
        albedo: Color::new(0.9, 0.95, 1.0), // casi blanco (afecta difuso)
        specular: 0.6,                      // brillos más fuertes
        reflectivity: 0.25,                 // reflejo medio
        transparency: 0.7,                  // bastante transparente
        ior: 1.33,                          // índice de refracción del agua
        tex: Texture::Solid(Color::new(0.2, 0.4, 0.9)), // azul agua
    };

    let wood = Material {
        albedo: Color::new(1.0,1.0,1.0),
        specular: 0.2,
        reflectivity: 0.00,
        transparency: 0.0,
        ior: 1.0,
        tex: Texture::Image(wood_img),
    };
    let leaves = Material {
        albedo: Color::new(1.0,1.0,1.0),
        specular: 0.2,
        reflectivity: 0.0,
        transparency: 0.0,
        ior: 1.0,
        tex: Texture::Image(leaves_img),
    };

  
    // --- Objetos ---
    let mut objs: Vec<Object> = Vec::new();


    // y = -2 base
    let map0: &[&str] = &[
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGSSSSSSSSSSGGGGGGG",
        "GGGGGSSSSSSSSSSSSSSGGGGG",
        "GGGGGSSSSSSSSSSSSSSGGGGG",
        "GGGGGSSSSSSSSSSSSSSGGGGG",
        "GGGGGSSSSSSSSSSSSSSGGGGG",
        "GGGGSGSSSSSSSSSSSGGGGGGG",
        "GGGGGGSSSSSSSSSGGGGGGGGG",
        "GGGGGGGGSSSSGGGGGGGGGGGG",
    ];
    // y = -1 base
    let map1: &[&str] = &[
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGSSSSSSSSSSGGGGGGG",
        "GGGGGSSSSSSSWWSSSSSGGGGG",
        "GGGGGSSSSSWWWWSSSSSGGGGG",
        "GGGGGSSSWWWWWWSSSSSGGGGG",
        "GGGGGSSWWWWWWWWSSSSGGGGG",
        "GGGGSGSSSWWWWSSSSGGGGGGG",
        "GGGGGGSSSSSSSSSGGGGGGGGG",
        "GGGGGGGGSSSSGGGGGGGGGGGG",
        "GGGGGGGGSSSSGGGGGGGGGGGG",
        "GGGGGGGGSSSSGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
    ];

    // y = 0
    let map2: &[&str] = &[
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGSSSSSSSGGGGGGGGG",
        "GGGGGGGSSSWWWWWSSGGGGGGG",
        "GGGGGGSSWWWWWWWWSSSGGGGG",
        "GGGGGSSWWWWWWWWWWSSGGGGG",
        "GGGGGSSWWWWWWWWWWSSGGGGG",
        "GGGGGSSSWWWWWWWWSSSGGGGG",
        "GGGGSGSSSWWWWWWSSGGGGGGG",
        "GGGGGGSSSWWWSSSGGGGGGGGG",
        "GGGGGGGGSSSSGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
    ];
    let map3: &[&str] = &[
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGGGGGGGGGGGGGGGGGGG",
        "GGGGGGG______GGGGGGGGGG",
        "GGGGGG__________GGGGGGGG",
        "GGGGG______________GGGGG",
        "GGG_______________SGGGGG",
        "GGG_______________SGGGGG",
        "GGG_________________GGGG",
        "GG___________________GGG",
        "GG____________________GG",
        "GGGG_________________GGG",
        "G____________________GGG",
        "GGGGG___________________",
        "GGGGGG__________________",
        "GGGGGGG_________________",
    ];
    let map4: &[&str] = &[
        "_________________GGGGGGG",
        "_MMM_______________GGGGG",
        "_M_M_______________GGGGG",
        "_M_M_________________GGG",
        "___________________GGGGG",
        "___________________GGGGG",
        "_____________________GGG",
        "______________________GG",
        "_______________________G",
        "______________________GG",
        "GGGG__________________GG",
        "GGGG____________________",
        "GGGGG___________________",
        "GGGGGG__________________",
    ];
    let map5: &[&str] = &[
        "________________________",
        "________________________",
        "_MMM________________M___",
        "_M_M____________________",
        "_M_M____________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "___M____________________",
        "________________________",
        "________________________",
    ];
    let map6: &[&str] = &[
        "________________________",
        "________________________",
        "_MMM________________M___",
        "_MMM____________________",
        "_MMM____________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "___M____________________",
        "________________________",
        "________________________",
    ];
    let map7: &[&str] = &[
        "________________________",
        "___________________LLL__",
        "___________________LLL__",
        "___________________LLL__",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "__LLL___________________",
        "__LLL___________________",
        "__LLL___________________",
        "________________________",
    ];
    let map8: &[&str] = &[
        "________________________",
        "____________________L___",
        "___________________LLL__",
        "____________________L___",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "________________________",
        "___L____________________",
        "__LLL___________________",
        "___L____________________",
        "________________________",
    ];
    

    // Construir el piso bloque a bloque
    for (rz,row) in map0.iter().enumerate() {
        for (rx,ch) in row.chars().enumerate() {
            let gx = rx as i32 - (map0[0].len()/2) as i32; // centrado en X
            let gz = rz as i32 - (map0.len()/2) as i32 - 5; // centrado en Z
            match ch {
                'G' => put(&mut objs, gx, -3, gz, &grass), // grama
                'S' => put(&mut objs, gx,-3, gz, &sand),  // arena
                'W' => put(&mut objs, gx, -3, gz, &water), // agua hundida
                _   => {}
            }
        }
    }
    // Construir el piso bloque a bloque
    for (rz,row) in map1.iter().enumerate() {
        for (rx,ch) in row.chars().enumerate() {
            let gx = rx as i32 - (map1[0].len()/2) as i32; // centrado en X
            let gz = rz as i32 - (map1.len()/2) as i32 - 5; // centrado en Z
            match ch {
                'G' => put(&mut objs, gx, -2, gz, &grass), // grama
                'S' => put(&mut objs, gx, -2, gz, &sand),  // arena
                'W' => put(&mut objs, gx, -2, gz, &water), // agua hundida
                'M' => put(&mut objs, gx, 3, gz, &wood),
                'L' => put(&mut objs, gx, 3, gz, &leaves),
                _   => {}
            }
        }
    }
    // Construir el piso bloque a bloque
    for (rz,row) in map2.iter().enumerate() {
        for (rx,ch) in row.chars().enumerate() {
            let gx = rx as i32 - (map2[0].len()/2) as i32; // centrado en X
            let gz = rz as i32 - (map2.len()/2) as i32 - 5; // centrado en Z
            match ch {
                'G' => put(&mut objs, gx, -1, gz, &grass), // grama
                'S' => put(&mut objs, gx, -1, gz, &sand),  // arena
                'W' => put(&mut objs, gx, -1, gz, &water), // agua hundida
                'M' => put(&mut objs, gx, -1, gz, &wood),
                'L' => put(&mut objs, gx, -1, gz, &leaves),
                _   => {}
            }
        }
    }
    // Construir el piso bloque a bloque
    for (rz,row) in map3.iter().enumerate() {
        for (rx,ch) in row.chars().enumerate() {
            let gx = rx as i32 - (map3[0].len()/2) as i32; // centrado en X
            let gz = rz as i32 - (map3.len()/2) as i32 - 5; // centrado en Z
            match ch {
                'G' => put(&mut objs, gx, 0, gz, &grass), // grama
                'S' => put(&mut objs, gx, 0, gz, &sand),  // arena
                'W' => put(&mut objs, gx, 0, gz, &water), // agua hundida
                'M' => put(&mut objs, gx, 0, gz, &wood),
                'L' => put(&mut objs, gx, 0, gz, &leaves),
                _   => {}
            }
        }
    }
    // Construir el piso bloque a bloque
    for (rz,row) in map4.iter().enumerate() {
        for (rx,ch) in row.chars().enumerate() {
            let gx = rx as i32 - (map4[0].len()/2) as i32; // centrado en X
            let gz = rz as i32 - (map4.len()/2) as i32 - 5; // centrado en Z
            match ch {
                'G' => put(&mut objs, gx, 1, gz, &grass), // grama
                'S' => put(&mut objs, gx, 1, gz, &sand),  // arena
                'W' => put(&mut objs, gx, 1, gz, &water), // agua hundida
                'M' => put(&mut objs, gx, 1, gz, &wood),
                'L' => put(&mut objs, gx, 1, gz, &leaves),
                _   => {}
            }
        }
    }
    
    for (rz,row) in map5.iter().enumerate() {
        for (rx,ch) in row.chars().enumerate() {
            let gx = rx as i32 - (map5[0].len()/2) as i32; // centrado en X
            let gz = rz as i32 - (map5.len()/2) as i32 - 5; // centrado en Z
            match ch {
                'G' => put(&mut objs, gx, 2, gz, &grass), // grama
                'S' => put(&mut objs, gx, 2, gz, &sand),  // arena
                'W' => put(&mut objs, gx, 2, gz, &water), // agua hundida
                'M' => put(&mut objs, gx, 2, gz, &wood),
                'L' => put(&mut objs, gx, 2, gz, &leaves),
                _   => {}
            }
        }
    }
    for (rz,row) in map6.iter().enumerate() {
        for (rx,ch) in row.chars().enumerate() {
            let gx = rx as i32 - (map6[0].len()/2) as i32; // centrado en X
            let gz = rz as i32 - (map6.len()/2) as i32 - 5; // centrado en Z
            match ch {
                'G' => put(&mut objs, gx, 3, gz, &grass), // grama
                'S' => put(&mut objs, gx, 3, gz, &sand),  // arena
                'W' => put(&mut objs, gx, 3, gz, &water), // agua hundida
                'M' => put(&mut objs, gx, 3, gz, &wood),
                'L' => put(&mut objs, gx, 3, gz, &leaves),
                _   => {}
            }
        }
    }
    for (rz,row) in map7.iter().enumerate() {
        for (rx,ch) in row.chars().enumerate() {
            let gx = rx as i32 - (map7[0].len()/2) as i32; // centrado en X
            let gz = rz as i32 - (map7.len()/2) as i32 - 5; // centrado en Z
            match ch {
                'G' => put(&mut objs, gx, 4, gz, &grass), // grama
                'S' => put(&mut objs, gx, 4, gz, &sand),  // arena
                'W' => put(&mut objs, gx, 4, gz, &water), // agua hundida
                'M' => put(&mut objs, gx, 4, gz, &wood),
                'L' => put(&mut objs, gx, 4, gz, &leaves),
                _   => {}
            }
        }
    }
    for (rz,row) in map8.iter().enumerate() {
        for (rx,ch) in row.chars().enumerate() {
            let gx = rx as i32 - (map8[0].len()/2) as i32; // centrado en X
            let gz = rz as i32 - (map8.len()/2) as i32 - 5; // centrado en Z
            match ch {
                'G' => put(&mut objs, gx, 5, gz, &grass), // grama
                'S' => put(&mut objs, gx, 5, gz, &sand),  // arena
                'W' => put(&mut objs, gx, 5, gz, &water), // agua hundida
                'M' => put(&mut objs, gx, 5, gz, &wood),
                'L' => put(&mut objs, gx, 5, gz, &leaves),
                _   => {}
            }
        }
    }


    // Luces
    let lights = vec![
        PointLight{ pos:Vec3::new(-10.0, 7.0,  2.0), color:Color::new(1.0,0.93,0.86), intensity:140.0 },
        PointLight{ pos:Vec3::new(  8.0, 5.0,  8.0), color:Color::new(0.86,0.92,1.0), intensity:110.0 },
    ];
    let sky = Sky::Gradient { top: Color::new(0.95,0.60,0.75), bottom: Color::new(0.05,0.10,0.20) };
    let scene = Scene { objects: objs, lights, sky };

    let frames = 120usize;
    create_dir_all("frames")?;
    let mut buffer = vec![Color::new(0.0,0.0,0.0); width*height];

    for i in 0..frames {
        let t = i as f32 / frames as f32;
        let ang = t * 2.0 * PI;
        let radius = 14.0;
        //camera settings
        let cam = Camera {
            pos: Vec3::new(ang.cos()*radius, 5.0, ang.sin()*radius - 8.0),
            target: Vec3::new(0.0, -1.0, -8.0),   
            up: Vec3::new(0.0,8.0,0.0),
            fov_deg: 90.0,
            aspect,
        };

        buffer.par_iter_mut().enumerate().for_each(|(idx, c)| {
            let x = (idx % width) as f32;
            let y = (idx / width) as f32;
            let ray = cam.ray_for(x, y, width as f32, height as f32);
            *c = scene.trace(&ray, 5);
        });

        let path = format!("frames/frame_{:03}.ppm", i);
        write_frame(&buffer, width, height, &path)?;
        eprintln!("Escribí {}", path);
    }

    Ok(())
}
